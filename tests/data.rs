extern crate blade;

use blade::data;
use blade::prelude::*;

#[test]
fn test_trees() {
    let dt = data::trees::load();
    assert_eq!(dt.data().rows(), 31);
    assert_eq!(dt.data().cols(), 2);

    assert_eq!(dt.target().size(), 31);
}
