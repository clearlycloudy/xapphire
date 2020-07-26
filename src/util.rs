use ndarray::prelude::*;
use num::Float;

const EPS: f64 = 1e-9;

pub fn arr2_eq(a: &Array<f64, Ix2>, b: &Array<f64, Ix2>) -> bool {
    if a.shape() != b.shape() {
        false
    } else {
        let (r, c) = (a.shape()[0], a.shape()[1]);
        for i in 0..r {
            for j in 0..c {
                if (a[[i, j]] - b[[i, j]]).abs() > EPS {
                    return false;
                }
            }
        }
        true
    }
}

pub fn arr1_eq(a: &Array<f64, Ix1>, b: &Array<f64, Ix1>) -> bool {
    if a.shape() != b.shape() {
        false
    } else {
        let r = a.shape()[0];
        for i in 0..r {
            if (a[i] - b[i]).abs() > EPS {
                dbg!(&a);
                dbg!(&b);
                dbg!((a[i] - b[i]).abs());
                return false;
            }
        }
        true
    }
}
