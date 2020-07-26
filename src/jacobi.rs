use ndarray::prelude::*;
use num::Float;

#[cfg(test)]
use crate::util::*;

pub fn solve_jacobi(a: & mut Array<f64, Ix2>,
                    b: &Array<f64, Ix1>,
                    eps: f64,
                    iter_max: usize) -> Result<(Array<f64, Ix1>,usize), & 'static str> {
        
    let mut x = Array::<f64, Ix1>::zeros(a.shape()[1]);
    let mut xx = Array::<f64, Ix1>::zeros(a.shape()[1]);
    let n = a.shape()[1];
    let mut it = 0;
    while it < iter_max {
        for k in 0..n{
            let mut v = 0.;
            for j in 0..n {
                if j != k {
                    v += a[[k,j]] * x[j];
                }
            }
            xx[k] = 1./a[[k,k]] * (b[k] - v);
        }
        let d = &x - &xx;
        if (&d*&d).sum().sqrt() < eps {
            break;
        }
        std::mem::swap(&mut x, &mut xx);
        it += 1;
    }
    Ok((x,it))
}

#[test]
fn test_jacobi(){
    let mut a = arr2(&[[10., -1., 2., 0.],
                       [-1., 11., -1., 3.],
                       [2., -1. , 10., -1.],
                       [0., 3., -1., 8.]]);

    let b = arr1(&[6., 25., -11., 15.]);

    let (x,_) = solve_jacobi(& mut a, &b, 1e-15, 100).expect("solve_jacobi");

    assert!(arr1_eq(&x, &arr1(&[1., 2., -1., 1.])));
}
