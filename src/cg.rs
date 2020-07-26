use ndarray::prelude::*;
use num::Float;

#[cfg(test)]
use crate::util::*;

pub fn solve_cg(
    a: &Array<f64, Ix2>,
    b: &Array<f64, Ix1>,
    x0: &Array<f64, Ix1>,
    eps: f64,
    iter_max: usize,
) -> Result<(Array<f64, Ix1>, usize, f64), &'static str> {

    if a.shape()[0] != a.shape()[1] {
        return Err("expect square matrix")
    }
    
    let n = a.shape()[0];
    
    if b.shape()[0] != n {
        return Err("dimension of b not matching a")
    }

    let mut r = b - &a.dot(x0);
    let r0 = r.clone();
    let mut d = r.clone();
    let mut alpha = 0.;
    let mut lambda : f64 = (&d * &r).sum() / (&d * &a.dot(&d)).sum();
    let mut x = x0 + &(lambda * &d);
    
    let mut it = 1;
    while it <= iter_max {
        let mut r_new = &r - &(a.dot(&d) * lambda);
        alpha = (&r_new * &r_new).sum() / (&r * &r).sum();
        d = &r_new + &(alpha * &d);
        lambda = (&d * &r_new).sum() / (&d * &a.dot(&d)).sum();
        x = x + lambda * &d;
        if (&r_new * &r_new).sum() / (&r0 * &r0).sum() < eps {
            break;
        }
        std::mem::swap(& mut r, & mut r_new);
        it += 1;
    }

    let d : Array<f64, Ix1> = b - &(a.dot(&x));
    let l2 = (&d*&d).sum();
    
    Ok((x, it, l2))
}

#[test]
fn test_cg_0(){
    let mut a = arr2(&[[4., 1.],
                       [1., 3.]]);

    let orig = a.clone();

    let b = arr1(&[1., 2.]);

    let x0 = arr1(&[2., 1.]);
    
    let (x, it, res) = solve_cg(&a, &b, &x0, 1e-9, 100).expect("solve_cg");
    dbg!(&x, it, res);
    assert!(arr1_eq(&x, &arr1(&[1./11., 7./11.])));
}


#[test]
fn test_cg_1(){
    let mut a = arr2(&[[5., 3., 1.],
                       [3., 4., 2.],
                       [1., 2., 3.]]);

    let orig = a.clone();

    let b = arr1(&[41., 47., 32.]);

    let x0 = arr1(&[0., 0., 0.]);
    
    let (x, it, res) = solve_cg(&a, &b, &x0, 1e-9, 100).expect("solve_cg");
    dbg!(&x, it, res);
    assert!(arr1_eq(&x, &arr1(&[3., 7., 5.])));
}
