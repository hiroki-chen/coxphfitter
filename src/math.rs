use ndarray::*;
use std::ops::Mul;

#[inline]
pub fn exp<S, D>(x: &ArrayBase<S, D>) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    x.mapv(|x_val| x_val.exp())
}

/// Calculates the outer product of two 1D array views.
///
/// The outer product of a vector `u` of length `m` and a vector `v` of length `n`
/// is an `m x n` matrix `M` where `M[i, j] = u[i] * v[j]`.
///
/// # Arguments
/// * `u`: A 1D array view (e.g., from an `Array1<T>`).
/// * `v`: A 1D array view (e.g., from an `Array1<T>`).
///
/// # Returns
/// An owned `Array2<T>` representing the outer product.
///
/// # Type Parameters
/// * `T`: The element type of the arrays. Must support `Copy` and `Mul` (multiplication),
///   where the output of multiplication is also `T`.
///
/// # Examples
/// ```
/// use ndarray::{arr1, arr2};
/// use my_crate::outer_product; // Assuming this function is in my_crate
///
/// let u_vec = arr1(&[1.0, 2.0]);
/// let v_vec = arr1(&[3.0, 4.0, 5.0]);
///
/// let result = outer_product(u_vec.view(), v_vec.view());
///
/// let expected = arr2(&[[3.0, 4.0, 5.0],
///                       [6.0, 8.0, 10.0]]);
/// assert_eq!(result, expected);
/// ```
pub fn outer_product<T>(u: ArrayView1<T>, v: ArrayView1<T>) -> Array2<T>
where
    T: Copy + Mul<Output = T>, // Elements must be copyable and multipliable
{
    // Reshape u to a column vector view: shape (m, 1)
    // Example: if u has shape (m,), u_col_view will have shape (m, 1)
    let u_col_view = u.insert_axis(Axis(1));

    // Reshape v to a row vector view: shape (1, n)
    // Example: if v has shape (n,), v_row_view will have shape (1, n)
    let v_row_view = v.insert_axis(Axis(0));

    // Perform element-wise multiplication with broadcasting.
    // (m, 1) * (1, n) will broadcast to (m, n).
    // The multiplication of two views (or a view and a reference) produces an owned Array.
    let result_matrix = &u_col_view * &v_row_view;

    result_matrix
}

/// Calculates softplus(x) = log(1 + exp(x)) in a numerically stable way.
pub fn stable_softplus_element(val: f64) -> f64 {
    // max(0, x) + log(1 + exp(-abs(x)))
    // which is equivalent to max(0, x) + ln_1p(exp(-abs(x)))
    // ln_1p(f) calculates ln(1 + f)
    val.max(0.0) + (-val.abs()).exp().ln_1p()
}

/// Calculates the "soft absolute value" element-wise for an ndarray.
///
/// Equivalent to Python: `1 / a * (logaddexp(0, -a * x) + logaddexp(0, a * x))`
/// which simplifies to `1 / a * (softplus(-a * x) + softplus(a * x))`.
///
/// # Arguments
/// * `x`: Input array.
/// * `a`: A scalar parameter controlling the "smoothness".
///        Typically `a > 0`. If `a == 0.0`, the function will panic as it leads to divergence.
///        If `a < 0.0`, the function behaves as `-soft_abs(x, -a)`.
///
/// # Returns
/// An array with the soft absolute value computed for each element of `x`.
///
/// # Panics
/// Panics if `a` is `0.0`.
pub fn soft_abs<S, D>(x: &ArrayBase<S, D>, a: f64) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if a == 0.0 {
        // The limit of soft_abs(x,a) as a -> 0+ for x != 0 is infinite ( ln(4)/a ).
        // For x = 0, softplus(0) = ln(2), so 2*ln(2)/a = ln(4)/a, also diverges.
        panic!(
            "Parameter 'a' cannot be zero for the soft_abs function, as it leads to divergence."
        );
    }

    // Apply the calculation element-wise using mapv
    let term1 = x.mapv(|x_val| stable_softplus_element(-a * x_val));
    let term2 = x.mapv(|x_val| stable_softplus_element(a * x_val));

    // (term1 + term2) results in a new owned Array
    // Then scalar multiplication
    (1.0 / a) * (term1 + term2)
}

/// Calculates the Elastic Net penalty.
///
/// Equivalent to Python:
/// `n * (penalizer * (l1_ratio * soft_abs(beta, a) + 0.5 * (1 - l1_ratio) * (beta**2))).sum()`
///
/// # Arguments
/// * `beta`: The coefficient array (typically 1D).
/// * `a`: The smoothness parameter for the `soft_abs` function.
/// * `n_samples`: A scaling factor, often the number of samples. Corresponds to 'n' in the Python lambda.
/// * `penalty_strength`: The overall strength of the penalty. Corresponds to 'self.penalizer'.
/// * `l1_ratio`: The Elastic Net mixing parameter (between 0 for L2 and 1 for L1). Corresponds to 'self.l1_ratio'.
///
/// # Returns
/// The calculated Elastic Net penalty as a single `f64` value.
///
/// # Panics
/// Panics if `a` is `0.0` (delegated to `soft_abs_ndarray`).
pub fn elastic_net_penalty<S, D>(
    beta: &ArrayBase<S, D>,
    a_smoothness: f64,
    n_samples: f64,
    penalty_strength: f64,
    l1_ratio: f64,
) -> f64
where
    S: Data<Elem = f64>,
    D: Dimension, // Typically Ix1 for a coefficient vector
{
    // 1. Calculate the L1-like component: l1_ratio * soft_abs(beta, a)
    // soft_abs_ndarray returns an owned Array<f64, D>
    let soft_abs_of_beta = soft_abs(beta, a_smoothness);
    let l1_component = l1_ratio * soft_abs_of_beta; // Element-wise scalar multiplication

    // 2. Calculate the L2-like component: 0.5 * (1 - l1_ratio) * (beta**2)
    let beta_squared = beta.mapv(|b_val| b_val * b_val); // Element-wise square
    let l2_component = 0.5 * (1.0 - l1_ratio) * beta_squared; // Element-wise scalar multiplication

    // 3. Combine L1 and L2 components: (l1_component + l2_component)
    // This results in an owned Array<f64, D>
    let combined_penalty_terms = l1_component + l2_component;

    // 4. Apply the overall penalty strength: penalizer * (...)
    // This results in an owned Array<f64, D>
    let scaled_penalty_terms = penalty_strength * combined_penalty_terms;

    // 5. Sum all terms in the resulting array: (...).sum()
    let sum_of_penalties = scaled_penalty_terms.sum(); // This returns an f64

    // 6. Final multiplication by n_samples: n * (...)
    n_samples * sum_of_penalties
}

/// This class abstracts complicated step size logic out of the fitters.
pub struct StepSizer {
    initial_step_size: f64,
    step_size: f64,
    temper_back_up: bool,
    norm_of_deltas: Vec<f64>,
}

impl StepSizer {
    #[inline]
    pub fn new(initial_step_size: f64) -> Self {
        Self {
            initial_step_size,
            step_size: initial_step_size,
            temper_back_up: false,
            norm_of_deltas: Vec::new(),
        }
    }

    pub fn update(&mut self, norm_of_delta: f64) {
        const SCALE: f64 = 1.3;
        const LOOKBACK: usize = 3;

        self.norm_of_deltas.push(norm_of_delta);

        if self.temper_back_up {
            self.step_size = self.initial_step_size.min(self.step_size * SCALE)
        }

        // Only allow small steps
        if norm_of_delta >= 15.0 {
            self.step_size *= 0.1;
            self.temper_back_up = true;
        } else if norm_of_delta > 5.0 && 15.0 > norm_of_delta {
            self.step_size *= 0.25;
            self.temper_back_up = true;
        }

        // recent non-monotonically decreasing is a concern
        let len = self.norm_of_deltas.len();
        let start_index = len.saturating_sub(LOOKBACK);

        if len >= LOOKBACK && !self.is_monotonically_decreasing(&self.norm_of_deltas[start_index..])
        {
            self.step_size *= 0.98;
        }

        // recent monotonically decreasing is good though
        if len >= LOOKBACK && self.is_monotonically_decreasing(&self.norm_of_deltas[start_index..])
        {
            self.step_size = (self.step_size * SCALE).min(1.0);
        }
    }

    fn is_monotonically_decreasing(&self, values: &[f64]) -> bool {
        if values.len() < 2 {
            return true; // A single value or empty slice is trivially monotonically decreasing
        }
        for i in 1..values.len() {
            if values[i] >= values[i - 1] {
                return false; // Found a pair that is not decreasing
            }
        }
        true // All pairs were decreasing
    }

    #[inline]
    pub fn next(&self) -> f64 {
        self.step_size
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Imports outer_product from the parent module
    use ndarray::{arr1, arr2, Array1}; // For creating test arrays // For float comparisons

    #[test]
    fn test_outer_product_f64() {
        let u_vec: Array1<f64> = arr1(&[1.0, 2.0, 3.0]);
        let v_vec: Array1<f64> = arr1(&[4.0, 5.0]);

        let result = outer_product(u_vec.view(), v_vec.view());
        let expected = arr2(&[
            [1.0 * 4.0, 1.0 * 5.0],
            [2.0 * 4.0, 2.0 * 5.0],
            [3.0 * 4.0, 3.0 * 5.0],
        ]);
        // Expected:
        // [[ 4.0,  5.0],
        //  [ 8.0, 10.0],
        //  [12.0, 15.0]]

        assert!(result.abs_diff_eq(&expected, 1e-9));
        assert_eq!(result.dim(), (3, 2));
    }

    #[test]
    fn test_outer_product_i32() {
        let u_vec = arr1(&[1, 2]);
        let v_vec = arr1(&[3, 4, 5]);

        let result = outer_product(u_vec.view(), v_vec.view());
        let expected = arr2(&[[1 * 3, 1 * 4, 1 * 5], [2 * 3, 2 * 4, 2 * 5]]);
        // Expected:
        // [[3,  4,  5],
        //  [6,  8, 10]]
        assert_eq!(result, expected);
        assert_eq!(result.dim(), (2, 3));
    }

    #[test]
    fn test_outer_product_single_elements() {
        let u_vec = arr1(&[10]);
        let v_vec = arr1(&[20]);

        let result = outer_product(u_vec.view(), v_vec.view());
        let expected = arr2(&[[200]]);

        assert_eq!(result, expected);
        assert_eq!(result.dim(), (1, 1));
    }

    #[test]
    fn test_outer_product_u_empty() {
        let u_vec: Array1<i32> = arr1(&[]);
        let v_vec = arr1(&[1, 2, 3]);

        let result = outer_product(u_vec.view(), v_vec.view());

        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 3);
        assert_eq!(result.len(), 0); // Total number of elements
    }

    #[test]
    fn test_outer_product_v_empty() {
        let u_vec = arr1(&[1, 2, 3]);
        let v_vec: Array1<i32> = arr1(&[]);

        let result = outer_product(u_vec.view(), v_vec.view());

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_outer_product_both_empty() {
        let u_vec: Array1<f64> = arr1(&[]);
        let v_vec: Array1<f64> = arr1(&[]);

        let result = outer_product(u_vec.view(), v_vec.view());

        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 0);
        assert_eq!(result.len(), 0);
    }
}
