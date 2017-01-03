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

    let expected = vec![4.837659653793274, 4.553851633474814, 4.816981265588829, 15.874115228921276,
                        19.869008437727466, 21.018326956518713, 16.19268807496156, 19.245949183164264,
                        21.413021404689722, 20.187581283767763, 22.01540227104848, 21.468464618616004,
                        21.468464618616004, 20.506154129808046, 23.95410968618176, 27.852202904652778,
                        31.583966481344966, 33.806481916796706, 30.60097760433255, 28.697035014921106,
                        34.388184394951004, 36.00831896404399, 35.3852597094808, 41.76899799551756,
                        44.87770231764653, 50.94286775764302, 52.22375109249125, 53.42851282520876,
                        53.89932887551053, 53.89932887551053, 68.51530482306924];
    assert_eq!(result.predicted(), &Vector::new(expected));
}

// ------------------------------------------------------------------
// Ridge
// ------------------------------------------------------------------

/*
#[test]
fn test_ridge_regression_outlier() {
    let mut model = RidgeRegressor::default();
    let inputs = matrix![1.; 2.; 3.; 4.; 5.];
    let targets = Vector::new(vec![10., 20., 30., 80., 50.]);
    let dt = SupervisedDataset(inputs, targets);

    model.train(&inputs, &targets).unwrap();

    let parameters = model.parameters().unwrap();
    assert_eq!(parameters, &Vector::new(vec![-0.18181818181820594, 12.727272727272734]));

    let mut model = RidgeRegressor::new(0.1);
    let inputs = matrix![1.; 2.; 3.; 4.; 5.];
    let targets = Vector::new(vec![10., 20., 30., 80., 50.]);
    model.train(&inputs, &targets).unwrap();
    let parameters = model.parameters().unwrap();
    assert_eq!(parameters, &Vector::new(vec![-3.5841584158415647, 13.861386138613856]));
}

#[test]
#[should_panic]
fn test_ridge_regression_invalid_alpha() {
    RidgeRegressor::new(-1.0);
}

#[test]
fn test_ridge_regression_datasets_trees() {
    use rm::datasets::trees;
    let trees = trees::load();

    let mut model = RidgeRegressor::default();
    model.train(&trees.data(), &trees.target()).unwrap();
    let params = model.parameters().unwrap();
    assert_eq!(params, &Vector::new(vec![-58.09806161950894, 4.68684745409343, 0.34441921086952676]));

    let predicted = model.predict(&trees.data()).unwrap();
    let expected = vec![4.9121170103334, 4.596075192213792, 4.844606261293432, 15.912019831077998, 19.949162219722425,
                        21.106685386870826, 16.18892829290755, 19.288701190733292, 21.479481990490267, 20.226070681551974,
                        22.072432270439435, 21.5078593832402, 21.5078593832402, 20.502979143381534, 23.975548644826723,
                        27.84929214264129, 31.637903462206083, 33.85706165471298, 30.565512473307443, 28.623262742630104,
                        34.38250118562216, 36.00870909817989, 35.34824806919077, 41.68968082859186, 44.81783111916752,
                        50.882355416739074, 52.16414411842728, 53.35004467832559, 53.818729423734936, 53.818729423734936,
                        68.41546728046455];
    assert_eq!(predicted, Vector::new(expected));
}

#[test]
fn test_ridge_regression_datasets_trees_alpha01() {
    use rm::datasets::trees;
    let trees = trees::load();

    let mut model = RidgeRegressor::new(0.1);
    model.train(&trees.data(), &trees.target()).unwrap();
    let params = model.parameters().unwrap();
    assert_eq!(params, &Vector::new(vec![-57.99878658933356, 4.706019761728981, 0.3397708268791373]));

    let predicted = model.predict(&trees.data()).unwrap();
    let expected = vec![4.84513531455659, 4.558087108679594, 4.819749407267118, 15.877920444118622, 19.877061838376648,
                        21.027205468307827, 16.19230536370829, 19.250242805620523, 21.419698916189105, 20.19144675796631,
                        22.02113204165577, 21.472421537191252, 21.472421537191252, 20.50583167755598, 23.956262567349498,
                        27.85190952602645, 31.589388621696962, 33.81156735326769, 30.597412854772223, 28.689619042791165,
                        34.38761457144487, 36.00836017754894, 35.38154114479282, 41.761029133628014, 44.871689196542405,
                        50.936792265787936, 52.21776704501286, 53.420633295946175, 53.891235272119076, 53.891235272119076,
                        68.50528244076838];
    assert_eq!(predicted, Vector::new(expected));
}

#[test]
fn test_ridge_regression_datasets_trees_alpha00() {
    // should be the same as LinRegressor
    use rm::datasets::trees;
    let trees = trees::load();

    let mut model = RidgeRegressor::new(0.0);
    model.train(&trees.data(), &trees.target()).unwrap();
    let params = model.parameters().unwrap();
    assert_eq!(params, &Vector::new(vec![-57.98765891838409, 4.708160503017506, 0.3392512342447438]));

    let predicted = model.predict(&trees.data()).unwrap();
    let expected = vec![4.837659653793278, 4.55385163347481, 4.816981265588826, 15.874115228921276,
                        19.869008437727473, 21.018326956518717, 16.192688074961563, 19.245949183164257,
                        21.413021404689726, 20.187581283767756, 22.015402271048487, 21.468464618616007,
                        21.468464618616007, 20.50615412980805, 23.954109686181766, 27.852202904652785,
                        31.583966481344966, 33.806481916796706, 30.60097760433255, 28.697035014921106,
                        34.388184394951004, 36.008318964043994, 35.38525970948079, 41.76899799551756,
                        44.87770231764652, 50.942867757643015, 52.223751092491256, 53.42851282520877,
                        53.899328875510534, 53.899328875510534, 68.51530482306926];
    assert_eq!(predicted, Vector::new(expected));
}
*/
