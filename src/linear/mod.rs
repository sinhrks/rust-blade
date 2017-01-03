use linalg::{BaseMatrix, Matrix, Vector};
use num::Float;

use dataset::{Dataset, SupervisedDataset};
use errors::{TrainingResult, PredictionResult};
use traits::{SupervisedTrainable, Predictable, RegressionEvaluable};


// ------------------------------------------------------------------
// Meta
// ------------------------------------------------------------------

pub enum Regularization<T> {
    None,
    L1(T),
    L2(T),
    L1L2(T, T),
}

// ------------------------------------------------------------------
// Trainer
// ------------------------------------------------------------------

pub struct LinearTrainer<T> {
    regularizer: Regularization<T>,
}

impl<T> Default for LinearTrainer<T> {
    fn default() -> Self {
        LinearTrainer { regularizer: Regularization::None }
    }
}

impl SupervisedTrainable<f64, f64> for LinearTrainer<f64> {
    type Model = LinearModel<f64>;

    fn fit(&self, data: &SupervisedDataset<f64, f64>) -> TrainingResult<Self::Model> {

        let ones = Matrix::<f64>::ones(data.len(), 1);
        let full = ones.hcat(data.data());
        let xt = full.transpose();
        let params = (&xt * full)
            .solve(&xt * data.target())
            .expect("Unable to solve linear equation.");

        let intercept = unsafe { *params.get_unchecked(0) };
        // ToDo: better way to spit?
        let coefs = Vector::new(params.into_iter().skip(1).collect::<Vec<f64>>());

        // let ones = Matrix::<f64>::ones(inputs.rows(), 1);
        // let full_inputs = ones.hcat(inputs);
        // let xt = full_inputs.transpose();
        // cancel regularization of intercept
        // let mut eye = Matrix::<f64>::identity(inputs.cols() + 1);
        // unsafe {
        // eye.get_unchecked_mut([0, 0]) = 0.
        // }
        // let left = &xt * full_inputs + eye * self.alpha;
        // let right = &xt * targets;
        // self.parameters = Some(left.solve(right).expect("Unable to solve linear equation."));
        // Ok(())
        //

        let lm = LinearModel {
            coefs: coefs,
            intercept: intercept,
        };
        Ok(lm)
    }
}

// ------------------------------------------------------------------
// Model
// ------------------------------------------------------------------

pub struct LinearModel<D> {
    coefs: Vector<D>,
    intercept: D,
}

impl<D> LinearModel<D> {
    pub fn coefs(&self) -> &Vector<D> {
        &self.coefs
    }

    pub fn intercept(&self) -> &D {
        &self.intercept
    }
}

impl Predictable<f64> for LinearModel<f64> {
    type Result = LinearResult<f64>;

    fn predict(&self, data: &Dataset<f64>) -> PredictionResult<Self::Result> {
        let predicted = data.data() * &self.coefs + self.intercept;
        let lr = LinearResult {
            actual: None,
            predicted: predicted,
        };
        Ok(lr)
    }
}

// ------------------------------------------------------------------
// Result
// ------------------------------------------------------------------

pub struct LinearResult<D> {
    actual: Option<Vector<D>>,
    predicted: Vector<D>,
}

impl<D> LinearResult<D> {
    pub fn actual(&self) -> &Vector<D> {
        if let &Some(ref actual) = &self.actual {
            actual
        } else {
            panic!("error");
        }
    }

    pub fn predicted(&self) -> &Vector<D> {
        &self.predicted
    }
}

#[cfg(test)]
mod tests {

    use linalg::{Matrix, Vector};
    use dataset::Dataset;

    #[test]
    fn test_data() {
        let m = matrix![1., 2.;
                        2., 3.];
        let t = Vector::new(vec![0, 1]);
        let d = Dataset::new(m);
        assert_eq!(d.len(), 2);

        let dt = d.set_target(t);
        assert_eq!(dt.len(), 2);
    }
}
