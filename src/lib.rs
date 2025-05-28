//! This crate implements a simple Cox proportional hazards model for survival analysis.

use anyhow::Result;
use polars::prelude::*;

/// The input arguments for the Cox proportional hazards model.
#[derive(Debug, Clone)]
pub struct CoxPHFitterArgs<'a> {
    /// Attach a penalty to the size of the coefficients during regression.
    /// This improves stability of the estimates and controls for high correlation between covariates.
    ///
    /// ```tex
    ///     $penalizer(\dfrac{1 - l1\_ratio}{2} || \beta ||^{2}_{2} + l1\_ratio ||\beta||_1 )$
    /// ```
    pub penalizer: f64,
    /// Specify how the fitter should estimate the baseline.
    pub baseline_estimation_method: &'a str,
    /// Specify what raio to assign to a L1 vs L2 penalty. Will default to 0.0 if not specified.
    pub l1_ratio: f64,
    /// The name of the column in DataFrame that contains the subjects’ lifetimes.
    pub duration_col: &'a str,
    /// The name of the column in DataFrame that contains the subjects’ death observation.
    pub event_col: &'a str,
    /// The columns to use as covariates in the model.
    pub robust: bool,
}

impl<'a> CoxPHFitterArgs<'a> {
    /// Creates a new instance of `CoxPHFitterArgs` with the provided parameters.
    pub fn new(
        penalizer: f64,
        baseline_estimation_method: &'a str,
        l1_ratio: f64,
        duration_col: &'a str,
        event_col: &'a str,
        robust: bool,
    ) -> Self {
        Self {
            penalizer,
            baseline_estimation_method,
            l1_ratio,
            duration_col,
            event_col,
            robust,
        }
    }

    pub fn set_penalizer(mut self, penalizer: f64) -> Self {
        self.penalizer = penalizer;
        self
    }

    pub fn set_baseline_estimation_method(mut self, method: &'a str) -> Self {
        self.baseline_estimation_method = method;
        self
    }

    pub fn set_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    pub fn set_duration_col(mut self, duration_col: &'a str) -> Self {
        self.duration_col = duration_col;
        self
    }

    pub fn set_event_col(mut self, event_col: &'a str) -> Self {
        self.event_col = event_col;
        self
    }

    pub fn set_robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }
}

impl<'a> Default for CoxPHFitterArgs<'a> {
    fn default() -> Self {
        Self::new(
            // Default values for the CoxPHFitterArgs
            0.0,        // penalizer
            "breslow",  // baseline_estimation_method
            0.0,        // l1_ratio
            "duration", // duration_col
            "event",    // event_col
            false,      // robust
        )
    }
}

/// This class implements fitting Cox’s proportional hazard model.
///
/// ```tex
///     $h(t \mid x) = h_0(t) \exp((x \overline{x})'\beta)$
/// ```
///
/// The baseline hazard can be modeled in two ways:
///
/// 1. (default) non-parametrically, using Breslow’s method. In this case, the entire model is the
///     traditional semi-parametric Cox model. Ties are handled using Efron’s method.
/// 2. parametrically, using a pre-specified number of cubic splines, or piecewise values. (on-going work)
#[derive(Debug, Clone)]
pub struct CoxPHFitter<'a> {
    args: CoxPHFitterArgs<'a>,
}

impl<'a> CoxPHFitter<'a> {
    /// Creates a new instance of `CoxPHFitter` with the provided arguments.
    #[inline]
    pub fn new(args: CoxPHFitterArgs<'a>) -> Self {
        Self { args }
    }

    /// Fit the Cox proportional hazard model to a dataset.
    pub fn fit(&self, df: &DataFrame) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const TEST_DF_PATH: &str = "./data/cox_data.parquet";

    #[test]
    fn test_fit() {
        let args = CoxPHFitterArgs::default()
            .set_event_col("event")
            .set_duration_col("T")
            .set_penalizer(0.1)
            .set_robust(true);
        let fitter = CoxPHFitter::new(args);

        // Load a test DataFrame (this is a placeholder, replace with actual DataFrame loading logic)
        let df = LazyFrame::scan_parquet(TEST_DF_PATH, Default::default())
            .unwrap()
            .collect()
            .unwrap();

        // Fit the Cox proportional hazards model
        assert!(fitter.fit(&df).is_ok());
    }
}
