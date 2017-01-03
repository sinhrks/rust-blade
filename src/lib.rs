extern crate num;
#[macro_use]
extern crate rulinalg;

pub mod linalg {
    pub use rulinalg::matrix::{Axes, Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix,
                               BaseMatrixMut};
    pub use rulinalg::vector::Vector;
}

/// basis
mod dataset;
mod errors;
mod traits;

/// learning models
pub mod linear;

/// sample data
pub mod data;

/// prelude
pub mod prelude;
