#[macro_use]
extern crate rulinalg;
extern crate blade;

use blade::prelude::*;

use blade::data;
use blade::linear::LinearTrainer;

// ------------------------------------------------------------------
// Linear
// ------------------------------------------------------------------

#[test]
fn test_linear_regression() {
    let data = Matrix::new(3, 1, vec![2.0, 3.0, 4.0]);
    let target = Vector::new(vec![5.0, 6.0, 7.0]);
    let d = Dataset::with_labels(data, target);

    let trainer = LinearTrainer::default();
    let model = trainer.fit(&d).unwrap();

    assert_eq!(model.coefs(), &Vector::new(vec![0.9999999999999973]));
    assert_eq!(model.intercept(), &3.000000000000009);
}

#[test]
fn test_linear_regression_outlier() {
    let data = matrix![1.; 2.; 3.; 4.; 5.];
    let target = Vector::new(vec![10., 20., 30., 80., 50.]);
    let d = Dataset::with_labels(data, target);

    let trainer = LinearTrainer::default();
    let model = trainer.fit(&d).unwrap();
    let parameters = model.coefs();

    assert_eq!(model.coefs(), &Vector::new(vec![14.000000000000002]));
    assert_eq!(model.intercept(), &-4.000000000000008);
}

#[test]
fn test_linear_regression_datasets_trees() {
    let trees = data::trees::load();

    let trainer = LinearTrainer::default();
    let model = trainer.fit(&trees).unwrap();

    let exp = Vector::new(vec![4.708160503017506, 0.3392512342447438]);
    assert_eq!(model.coefs(), &exp);
    assert_eq!(model.intercept(), &-57.98765891838409);

    let result = model.predict(&trees.strip()).unwrap();

    let expected = vec![4.837659653793274,
                        4.553851633474814,
                        4.816981265588829,
                        15.874115228921276,
                        19.869008437727466,
                        21.018326956518713,
                        16.19268807496156,
                        19.245949183164264,
                        21.413021404689722,
                        20.187581283767763,
                        22.01540227104848,
                        21.468464618616004,
                        21.468464618616004,
                        20.506154129808046,
                        23.95410968618176,
                        27.852202904652778,
                        31.583966481344966,
                        33.806481916796706,
                        30.60097760433255,
                        28.697035014921106,
                        34.388184394951004,
                        36.00831896404399,
                        35.3852597094808,
                        41.76899799551756,
                        44.87770231764653,
                        50.94286775764302,
                        52.22375109249125,
                        53.42851282520876,
                        53.89932887551053,
                        53.89932887551053,
                        68.51530482306924];
    assert_eq!(result.predicted(), &Vector::new(expected));
}
