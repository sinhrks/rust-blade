use dataset::{Dataset, SupervisedDataset};
use errors::{TrainingResult, PredictionResult};

/// trait for supervised Trainer, which outputs Model
pub trait SupervisedTrainable<D, T> {
    type Model;

    fn fit(&self, data: &SupervisedDataset<D, T>) -> TrainingResult<Self::Model>;
}

/// trait for Trainer, which outputs Model
pub trait UnsupervisedTrainable<D> {
    type Model;

    fn fit(&self, data: &Dataset<D>) -> Self::Model;
}

/// trait for Model, which outputs Result
pub trait Predictable<D> {
    type Result;

    fn predict(&self, data: &Dataset<D>) -> PredictionResult<Self::Result>;
}

/// trait for Result
pub trait Evaluable<T> {
    fn actual(&self) -> T;
    fn predicted(&self) -> T;
}

pub trait RegressionEvaluable<T> : Evaluable<T> {

}

pub trait ClassificationEvaluable<T> : Evaluable<T> {

}

pub trait ClusteringEvaluable<T> : Evaluable<T> {

}
