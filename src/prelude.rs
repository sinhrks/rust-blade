//! The blade prelude.

pub use linalg::{Matrix, MatrixSlice, MatrixSliceMut,
                 BaseMatrix, BaseMatrixMut, Vector, Axes};

pub use dataset::Dataset;
pub use traits::{SupervisedTrainable, UnsupervisedTrainable, Predictable,
                 Evaluable, RegressionEvaluable,
                 ClassificationEvaluable, ClusteringEvaluable};
