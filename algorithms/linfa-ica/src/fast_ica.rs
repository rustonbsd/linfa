//! Fast algorithm for Independent Component Analysis (ICA)

use linfa::{
    dataset::{DatasetBase, Records, WithLapack, WithoutLapack},
    traits::*,
    Float,
};
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::{eigh::Eigh, solveh::UPLO, svd::SVD};
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{StandardNormal, Uniform},
    RandomExt,
};
use num_traits::ToPrimitive as _;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use crate::error::{FastIcaError, Result};
use crate::hyperparams::FastIcaValidParams;

// Simple QR decomposition using Gram-Schmidt
fn qr_decomposition<F: Float>(a: &Array2<F>) -> Result<(Array2<F>, Array2<F>)> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut q = Array2::<F>::zeros((m, n));
    let mut r = Array2::<F>::zeros((n, n));

    for j in 0..n {
        let mut v = a.column(j).to_owned();

        // Orthogonalize against previous columns
        for i in 0..j {
            let q_i = q.column(i);
            let r_ij = v.dot(&q_i);
            r[[i, j]] = r_ij;
            v.scaled_add(-r_ij, &q_i);
        }

        // Normalize
        let norm = v
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt();
        if norm > F::cast(1e-10) {
            r[[j, j]] = norm;
            q.column_mut(j).assign(&v.mapv(|x| x / norm));
        }
    }

    Ok((q, r))
}

#[cfg(feature = "blas")]
impl<F: Float + ndarray_linalg::Lapack, D: Data<Elem = F>, T> Fit<ArrayBase<D, Ix2>, T, FastIcaError>
    for FastIcaValidParams<F>
{
    type Object = FastIca<F>;

    /// Fit the model
    ///
    /// # Errors
    ///
    /// If the [`FastIcaValidParams::ncomponents`] is set to a number greater than the minimum of
    /// the number of rows and columns
    ///
    /// If the `alpha` value set for [`GFunc::Logcosh`] is not between 1 and 2
    /// inclusive
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let x = &dataset.records;
        let (nsamples, nfeatures) = (x.nsamples(), x.nfeatures());
        if dataset.nsamples() == 0 {
            return Err(FastIcaError::NotEnoughSamples);
        }

        // If the number of components is not set, we take the minimum of
        // the number of rows and columns
        let ncomponents = self
            .ncomponents()
            .unwrap_or_else(|| nsamples.min(nfeatures));

        // The number of components cannot be greater than the minimum of
        // the number of rows and columns
        if ncomponents > nsamples.min(nfeatures) {
            return Err(FastIcaError::InvalidValue(format!(
                "ncomponents cannot be greater than the min({nsamples}, {nfeatures}), got {ncomponents}"
            )));
        }

        println!(
            "[FastICA] Starting fit: {} samples, {} features, {} components",
            nsamples, nfeatures, ncomponents
        );

        // We center the input by subtracting the mean of its features
        // safe unwrap because we already returned an error on zero samples
        println!("[FastICA] Computing mean and centering data...");
        let xmean = x.mean_axis(Axis(0)).unwrap();
        let mut xcentered = x - &xmean.view().insert_axis(Axis(0));

        // We transpose the centered matrix
        println!("[FastICA] Transposing centered matrix...");
        xcentered = xcentered.reversed_axes();

        // We whiten the matrix to remove any potential correlation between
        // the components
        println!(
            "[FastICA] Computing SVD for whitening... (matrix size: {}x{})",
            xcentered.nrows(),
            xcentered.ncols()
        );

        // Use randomized SVD for large matrices (> 10000 elements or if ncomponents << min(dims))
        let use_randomized = true; //xcentered.len() > 100_000_000 || (ncomponents < nsamples.min(nfeatures) / 4);

        let k = if use_randomized {
            println!("[FastICA] Using randomized SVD for large-scale whitening");
            let n_oversamples = (ncomponents / 2).min(nsamples.min(nfeatures) - ncomponents);
            let n_iter = 15; // Number of power iterations
            let seed = self.random_state().map(|s| s as u64);
            let (u, s) = Self::randomized_svd(
                &xcentered.view().to_owned(),
                ncomponents,
                n_oversamples,
                n_iter,
                seed,
            )?;
            (u / &s.insert_axis(Axis(0))).t().to_owned()
        } else {
            println!("[FastICA] Using standard SVD");
            let xcentered_lapack = xcentered.view().with_lapack();
            match xcentered_lapack.svd(true, false)? {
                (Some(u), s, _) => {
                    let s = s.mapv(F::Lapack::cast);
                    (u.slice_move(s![.., ..nsamples.min(nfeatures)]) / s)
                        .t()
                        .slice(s![..ncomponents, ..])
                        .to_owned()
                        .without_lapack()
                }
                _ => return Err(FastIcaError::SvdDecomposition),
            }
        };

        println!("[FastICA] Whitening data...");
        let mut xwhitened = k.dot(&xcentered);

        // We multiply the matrix with root of the number of records
        let nsamples_sqrt = num_traits::Float::sqrt(F::cast(nsamples));
        xwhitened.mapv_inplace(|x| x * nsamples_sqrt);

        // We initialize the de-mixing matrix with a uniform distribution
        println!("[FastICA] Initializing de-mixing matrix...");
        let w: Array2<f64>;
        if let Some(seed) = self.random_state() {
            let mut rng = Xoshiro256Plus::seed_from_u64(*seed as u64);
            w = Array::random_using((ncomponents, ncomponents), Uniform::new(0., 1.), &mut rng);
        } else {
            w = Array::random((ncomponents, ncomponents), Uniform::new(0., 1.));
        }
        let mut w = w.mapv(F::cast);

        // We find the optimized de-mixing matrix
        println!("[FastICA] Running parallel ICA optimization...");
        w = self.ica_parallel(&xwhitened, &w)?;

        // We whiten the de-mixing matrix
        println!("[FastICA] Computing final components...");
        let components = w.dot(&k);

        println!("[FastICA] Fit complete!");
        Ok(FastIca {
            mean: xmean,
            components,
        })
    }
}

#[cfg(feature = "blas")]
impl<F: Float + ndarray_linalg::Lapack> FastIcaValidParams<F> {
    // Randomized SVD for large-scale whitening
    // Based on Halko, Martinsson, and Tropp (2011)
    fn randomized_svd(
        x: &Array2<F>, // shape: D x N (features x samples)
        n_components: usize,
        n_oversamples: usize,
        n_iter: usize,
        random_state: Option<u64>,
    ) -> Result<(Array2<F>, Array1<F>)> {
        let (d, n) = (x.nrows(), x.ncols());
        let mut r = n_components + n_oversamples.max(32);
        r = r.min(d).min(n);

        println!(
            "[RandomizedSVD] D={} N={} r={} (k={} + p={}), iters={}",
            d,
            n,
            r,
            n_components,
            r.saturating_sub(n_components),
            n_iter
        );

        // Omega: N x r
        let omega: Array2<F> = if let Some(seed) = random_state {
            let mut rng = Xoshiro256Plus::seed_from_u64(seed);
            Array::random_using((n, r), StandardNormal, &mut rng).mapv(|z: f64| F::cast(z))
        } else {
            Array::random((n, r), StandardNormal).mapv(|z: f64| F::cast(z))
        };

        // Y = X * Omega (D x r), with power iterations and re-orth via thin SVD
        let mut y = x.dot(&omega);
        for i in 0..n_iter {
            println!("[RandomizedSVD] power iter {}/{}", i + 1, n_iter);
            // Z = X^T * Y (N x r) then Y = X * Z (D x r)
            let z = x.t().dot(&y);
            y = x.dot(&z);

            // Re-orthonormalize Y -> Q via thin SVD for stability
            let (u_opt, _s_opt, _vt_opt) = y.with_lapack().svd(true, false)?;
            y = u_opt
                .ok_or(FastIcaError::SvdDecomposition)?
                .without_lapack();
        }

        // Final Q from thin SVD of Y (stable)
        let (u_opt, _s_opt, _vt_opt) = y.with_lapack().svd(true, false)?;
        let q = u_opt
            .ok_or(FastIcaError::SvdDecomposition)?
            .slice_move(s![.., ..r])
            .without_lapack(); // D x r

        // Stream over X by column blocks to build M = (Q^T X)(Q^T X)^T (r x r)
        let mut m = Array2::<F>::zeros((r, r));
        let block = 4096usize.max(2 * r).min(n); // tuneable; keeps cache/mem happy

        println!("[RandomizedSVD] accumulating M with block={}", block);
        let mut start = 0usize;
        while start < n {
            let end = (start + block).min(n);
            let x_blk = x.slice(s![.., start..end]); // D x b
            let z = q.t().dot(&x_blk); // r x b
                                       // M += Z Z^T
            m = m + z.dot(&z.t());
            start = end;
        }

        // Eigendecompose M = V diag(s^2) V^T
        let (eig_vals_raw, v_raw) = {
            #[cfg(feature = "blas")]
            {
                use ndarray_linalg::eigh::Eigh;
                m.with_lapack().eigh(UPLO::Upper)?
            }
            #[cfg(not(feature = "blas"))]
            {
                m.eigh()?
            }
        };
        let mut s = eig_vals_raw.mapv(|lam| {
            let z = F::cast(lam.to_f64().unwrap().max(0.0));
            num_traits::Float::sqrt(z)
        });
        let v = v_raw.without_lapack(); // r x r

        // Sort by descending singular value
        let mut idx: Vec<usize> = (0..r).collect();
        idx.sort_by(|&i, &j| s[[j]].partial_cmp(&s[[i]]).unwrap());
        let take = n_components.min(r);

        let mut u = Array2::<F>::zeros((d, take));
        let mut s_k = Array1::<F>::zeros(take);
        for (t, &i) in idx.iter().take(take).enumerate() {
            let v_i = v.column(i).to_owned(); // r
            u.column_mut(t).assign(&q.dot(&v_i)); // D
            s_k[t] = s[i];
        }

        Ok((u, s_k))
    }

    fn ica_parallel(&self, x: &Array2<F>, w_init: &Array2<F>) -> Result<Array2<F>> {
        println!(
            "[ICA] Starting parallel optimization (max_iter: {})",
            self.max_iter()
        );
        let mut w = Self::sym_decorrelation(w_init)?;

        let p = F::cast(x.ncols() as f64);
        let tol = F::cast(self.tol());
        let max_iter = self.max_iter();

        for iter in 0..max_iter {
            let wtx = w.dot(x);

            let (gwtx, g_wtx) = self.gfunc().exec(&wtx)?;

            let mut lhs = gwtx.dot(&x.t());
            lhs.par_mapv_inplace(|v| v / p);

            let rhs = {
                let mut out = w.clone();
                let g_wtx_vec: Vec<F> = g_wtx.iter().cloned().collect();
                out.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(g_wtx_vec.into_par_iter())
                    .for_each(|(mut row, gi)| {
                        for v in row.iter_mut() {
                            *v *= gi;
                        }
                    });
                out
            };

            let mut delta = lhs;
            delta
                .iter_mut()
                .zip(rhs.iter())
                .par_bridge()
                .for_each(|(d, r)| {
                    *d -= *r;
                });

            let wnew = Self::sym_decorrelation(&delta)?;

            let lim = wnew
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(w.axis_iter(Axis(0)).into_par_iter())
                .map(|(wn_row, w_row)| {
                    let dot = wn_row.dot(&w_row);
                    num_traits::Float::abs(num_traits::Float::abs(dot) - F::cast(1.))
                })
                .reduce(|| F::cast(0.), |a, b| if a > b { a.max(b) } else { b });

            w = wnew;

            if lim < tol {
                println!(
                    "[ICA] Converged at iteration {} (lim: {:e} < tol: {:e})",
                    iter + 1,
                    lim,
                    tol
                );
                break;
            }
        }

        println!("[ICA] Parallel optimization complete");
        Ok(w)
    }

    fn sym_decorrelation(w: &Array2<F>) -> Result<Array2<F>> {
        println!("[ICA] Starting symmetric decorrelation...");
        let s = w.dot(&w.t()).with_lapack();

        #[cfg(feature = "blas")]
        let (eig_val_raw, eig_vec_raw) = s.eigh(UPLO::Upper)?;
        #[cfg(not(feature = "blas"))]
        let (eig_val_raw, eig_vec_raw) = s.eigh()?;

        let eig_val = eig_val_raw.mapv(F::cast);
        let eig_vec = eig_vec_raw.without_lapack();

        let inv_sqrt = {
            let mut out = eig_val.clone();
            out.par_mapv_inplace(|lambda| {
                let mut s = num_traits::Float::sqrt(lambda);
                let lower = F::cast(1e-7);
                if s < lower {
                    s = lower;
                }
                s.recip()
            });
            out
        };

        let mut tmp = eig_vec.clone();
        tmp.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for (v, scale) in row.iter_mut().zip(inv_sqrt.iter()) {
                    *v *= *scale;
                }
            });

        println!("[ICA] Symmetric decorrelation complete");
        Ok(tmp.dot(&eig_vec.t()).dot(w))
    }
}

/// Fitted FastICA model for recovering the sources
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct FastIca<F> {
    mean: Array1<F>,
    components: Array2<F>,
}

impl<F: Float> FastIca<F> {
    pub fn components(&self) -> &Array2<F> {
        &self.components
    }

    pub fn mean(&self) -> &Array1<F> {
        &self.mean
    }
}

impl<F: Float> PredictInplace<Array2<F>, Array2<F>> for FastIca<F> {
    /// Recover the sources
    fn predict_inplace(&self, x: &Array2<F>, y: &mut Array2<F>) {
        assert_eq!(
            y.shape(),
            &[x.nrows(), self.components.nrows()],
            "The number of data points must match the number of output targets."
        );

        let xcentered = x - &self.mean.view().insert_axis(Axis(0));
        *y = xcentered.dot(&self.components.t());
    }

    fn default_target(&self, x: &Array2<F>) -> Array2<F> {
        Array2::zeros((x.nrows(), self.components.nrows()))
    }
}

/// Some standard non-linear functions
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum GFunc {
    Logcosh(f64),
    Exp,
    Cube,
}

impl GFunc {
    // Function to select the correct non-linear function and execute it
    // returning a tuple, consisting of the first and second derivatives of the
    // non-linear function
    fn exec<A: Float>(&self, x: &Array2<A>) -> Result<(Array2<A>, Array1<A>)> {
        match self {
            Self::Cube => Ok(Self::cube(x)),
            Self::Exp => Ok(Self::exp(x)),
            Self::Logcosh(alpha) => Self::logcosh(x, *alpha),
        }
    }

    fn cube<A: Float>(x: &Array2<A>) -> (Array2<A>, Array1<A>) {
        let gwtx = {
            let mut out = x.clone();
            out.par_mapv_inplace(|v| v.powi(3));
            out
        };

        let g_wtx = {
            let nrows = x.nrows();
            let mut out = Array1::<A>::zeros(nrows);
            out.iter_mut().enumerate().par_bridge().for_each(|(i, o)| {
                let row = x.row(i);
                let mut acc = A::cast(0.);
                let len = A::cast(row.len() as f64);
                for v in row.iter() {
                    let v2 = *v * *v;
                    acc += A::cast(3.) * v2;
                }
                *o = acc / len;
            });
            out
        };

        (gwtx, g_wtx)
    }

    fn exp<A: Float>(x: &Array2<A>) -> (Array2<A>, Array1<A>) {
        let mut exp_term = x.clone();
        exp_term.par_mapv_inplace(|v| {
            let half = A::cast(0.5);
            (-(v * v) * half).exp()
        });

        let mut gwtx = x.clone();
        gwtx.iter_mut()
            .zip(exp_term.iter())
            .par_bridge()
            .for_each(|(g, e)| {
                *g *= *e;
            });

        let g_wtx = {
            let mut out = Array1::<A>::zeros(x.nrows());
            out.iter_mut().enumerate().par_bridge().for_each(|(i, o)| {
                let row_x = x.row(i);
                let row_e = exp_term.row(i);
                let mut acc = A::cast(0.);
                let len = A::cast(row_x.len() as f64);
                for (vx, ve) in row_x.iter().zip(row_e.iter()) {
                    let one = A::cast(1.);
                    let term = (one - (*vx * *vx)) * *ve;
                    acc += term;
                }
                *o = acc / len;
            });
            out
        };

        (gwtx, g_wtx)
    }

    fn logcosh<A: Float>(x: &Array2<A>, alpha: f64) -> Result<(Array2<A>, Array1<A>)> {
        if !(1.0..=2.0).contains(&alpha) {
            return Err(FastIcaError::InvalidValue(format!(
                "alpha must be between 1 and 2 inclusive, got {alpha}"
            )));
        }
        let alpha_a = A::cast(alpha);

        let mut gx = x.clone();
        gx.par_mapv_inplace(|v| (v * alpha_a).tanh());

        let mut g_x = gx.clone();
        g_x.par_mapv_inplace(|v| {
            let one = A::cast(1.);
            alpha_a * (one - (v * v))
        });

        let g_wtx = {
            let mut out = Array1::<A>::zeros(g_x.nrows());
            out.iter_mut().enumerate().par_bridge().for_each(|(i, o)| {
                let row = g_x.row(i);
                let mut acc = A::cast(0.);
                let len = A::cast(row.len() as f64);
                for v in row.iter() {
                    acc += *v;
                }
                *o = acc / len;
            });
            out
        };

        Ok((gx, g_wtx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::traits::{Fit, Predict};

    use crate::hyperparams::{FastIcaParams, FastIcaValidParams};
    use ndarray_rand::rand_distr::StudentT;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<FastIca<f64>>();
        has_autotraits::<GFunc>();
        has_autotraits::<FastIcaParams<f64>>();
        has_autotraits::<FastIcaValidParams<f64>>();
        has_autotraits::<FastIcaError>();
    }

    // Test to make sure the number of components set cannot be greater
    // that the minimum of the number of rows and columns of the input
    #[test]
    fn test_ncomponents_err() {
        let input = DatasetBase::from(Array::random((4, 4), Uniform::new(0.0, 1.0)));
        let ica = FastIca::params().ncomponents(100);
        let ica = ica.fit(&input);
        assert!(ica.is_err());
    }

    // Test to make sure the alpha value of the `GFunc::Logcosh` is between
    // 1 and 2 inclusive
    #[test]
    fn test_logcosh_alpha_err() {
        let input = DatasetBase::from(Array::random((4, 4), Uniform::new(0.0, 1.0)));
        let ica = FastIca::params().gfunc(GFunc::Logcosh(10.));
        let ica = ica.fit(&input);
        assert!(ica.is_err());
    }

    // Helper macro that produces test-cases with the pattern test_fast_ica_*
    macro_rules! fast_ica_tests {
        ($($name:ident: $gfunc:expr,)*) => {
            paste::item! {
                $(
                    #[test]
                    fn [<test_fast_ica_$name>]() {
                        test_fast_ica($gfunc);
                    }
                )*
            }
        }
    }

    // Tests to make sure all of the `GFunc`'s non-linear functions and the
    // model itself performs well
    fast_ica_tests! {
        exp: GFunc::Exp, cube: GFunc::Cube, logcosh: GFunc::Logcosh(1.0),
    }

    // Helper function that mixes two signal sources sends it to FastICA
    // and makes sure the model can demix them with considerable amount of
    // accuracy
    fn test_fast_ica(gfunc: GFunc) {
        let nsamples = 1000;

        // Center the data and make it have unit variance
        let center_and_norm = |s: &mut Array2<f64>| {
            let mean = s.mean_axis(Axis(0)).unwrap();
            *s -= &mean.insert_axis(Axis(0));
            let std = s.std_axis(Axis(0), 0.);
            *s /= &std.insert_axis(Axis(0));
        };

        // Creaing a sawtooth signal
        let mut source1 = Array::linspace(0., 100., nsamples);
        source1.mapv_inplace(|x| {
            let tmp = 2. * f64::sin(x);
            if tmp > 0. {
                return 0.;
            }
            -1.
        });

        // Creating noise using Student T distribution
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let source2 = Array::random_using((nsamples, 1), StudentT::new(1.0).unwrap(), &mut rng);

        // Column concatenating both the sources
        let mut sources = concatenate![Axis(1), source1.insert_axis(Axis(1)), source2];
        center_and_norm(&mut sources);

        // Mixing the two sources
        let phi: f64 = 0.6;
        let mixing = array![[phi.cos(), phi.sin()], [phi.sin(), -phi.cos()]];
        sources = mixing.dot(&sources.t());
        center_and_norm(&mut sources);

        sources = sources.reversed_axes();

        // We fit and transform using the model to unmix the two sources
        let ica = FastIca::params()
            .ncomponents(2)
            .gfunc(gfunc)
            .random_state(42);

        let sources_dataset = DatasetBase::from(sources.view());
        let ica = ica.fit(&sources_dataset).unwrap();
        let mut output = ica.predict(&sources);

        center_and_norm(&mut output);

        // Making sure the model output has the right shape
        assert_eq!(output.shape(), &[1000, 2]);

        // The order of the sources in the ICA output is not deterministic,
        // so we account for that here
        let s1 = sources.column(0);
        let s2 = sources.column(1);
        let mut s1_ = output.column(0);
        let mut s2_ = output.column(1);
        if s1_.dot(&s2).abs() > s1_.dot(&s1).abs() {
            s1_ = output.column(1);
            s2_ = output.column(0);
        }

        let similarity1 = s1.dot(&s1_).abs() / (nsamples as f64);
        let similarity2 = s2.dot(&s2_).abs() / (nsamples as f64);

        // We make sure the saw tooth signal identified by ICA using the mixed
        // source is similar to the original sawtooth signal
        // We ignore the noise signal's similarity measure
        assert!(similarity1.max(similarity2) > 0.9);
    }
}
