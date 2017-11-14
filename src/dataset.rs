use linalg::{BaseMatrix, Matrix, Vector};

pub struct Dataset<D> {
    data: Matrix<D>,
}

impl<D> Dataset<D> {
    pub fn new(data: Matrix<D>) -> Self {
        Dataset { data: data }
    }

    pub fn with_labels<T>(data: Matrix<D>, target: Vector<T>) -> SupervisedDataset<D, T> {
        SupervisedDataset {
            data: data,
            target: target,
        }
    }

    pub fn len(&self) -> usize {
        self.data.rows()
    }

    pub fn data(&self) -> &Matrix<D> {
        &self.data
    }

    pub fn set_target<T>(self, target: Vector<T>) -> SupervisedDataset<D, T> {
        SupervisedDataset {
            data: self.data,
            target: target,
        }
    }
}

pub struct SupervisedDataset<D, T> {
    data: Matrix<D>,
    target: Vector<T>,
}

impl<D, T> SupervisedDataset<D, T> {
    pub fn new(data: Matrix<D>, target: Vector<T>) -> Self {
        SupervisedDataset {
            data: data,
            target: target,
        }
    }

    // temp
    pub fn strip(self) -> Dataset<D> {
        Dataset { data: self.data }
    }

    pub fn len(&self) -> usize {
        self.data.rows()
    }

    pub fn data(&self) -> &Matrix<D> {
        &self.data
    }

    pub fn target(&self) -> &Vector<T> {
        &self.target
    }
}

#[cfg(test)]
mod tests {

    use linalg::{Matrix, Vector};
    use super::Dataset;

    #[test]
    fn test_data() {
        let m =
            matrix![1., 2.;
                        2., 3.];
        let t = Vector::new(vec![0, 1]);
        let d = Dataset::new(m);
        assert_eq!(d.len(), 2);

        let dt = d.set_target(t);
        assert_eq!(dt.len(), 2);
    }
}
