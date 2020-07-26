use ndarray::prelude::*;
use num::Float;

#[cfg(test)]
use crate::util::*;

pub fn solve_ssor(
    a: &mut Array<f64, Ix2>,
    b: &Array<f64, Ix1>,
    w: f64,
    eps: f64,
    iter_max: usize,
) -> Result<(Array<f64, Ix1>, usize), &'static str> {
    let mut x0 = Array::<f64, Ix1>::zeros(a.shape()[1]);
    let mut x1 = Array::<f64, Ix1>::zeros(a.shape()[1]);
    let mut x2 = Array::<f64, Ix1>::zeros(a.shape()[1]);
    let n = a.shape()[1];
    let mut it = 0;
    while it < iter_max {
        for k in 0..n {
            let mut v = 0.;
            for j in 0..k {
                v += a[[k, j]] * x1[j];
            }
            for j in k + 1..n {
                v += a[[k, j]] * x0[j];
            }
            x1[k] = (1. - w) * x0[k] + w / a[[k, k]] * (b[k] - v);
        }

        for k in (0..n).rev() {
            let mut v = 0.;
            for j in 0..k {
                v += a[[k, j]] * x2[j];
            }
            for j in k + 1..n {
                v += a[[k, j]] * x1[j];
            }
            x2[k] = (1. - w) * x1[k] + w / a[[k, k]] * (b[k] - v);
        }

        let d = &x2 - &x0;
        if (&d * &d).sum().sqrt() < eps {
            break;
        }
        std::mem::swap(&mut x2, &mut x0);
        it += 1;
    }
    Ok((x2, it))
}

#[test]
fn test_ssor() {
    let mut a = arr2(&[
        [10., -1., 2., 0.],
        [-1., 11., -1., 3.],
        [2., -1., 10., -1.],
        [0., 3., -1., 8.],
    ]);

    let b = arr1(&[6., 25., -11., 15.]);

    let (x, it) = solve_ssor(&mut a, &b, 1.0, 1e-15, 100).expect("solve_ssor");
    dbg!(&x, it);
    assert!(arr1_eq(&x, &arr1(&[1., 2., -1., 1.])));
}
