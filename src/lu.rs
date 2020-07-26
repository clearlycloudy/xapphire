use ndarray::prelude::*;
use num::Float;

#[cfg(test)]
use crate::util::*;

pub fn solve_lu(
    a: &mut Array<f64, Ix2>,
    b: &Array<f64, Ix1>,
) -> Result<Array<f64, Ix1>, &'static str> {
    
    if a.shape()[0] != b.shape()[0] {
        return Err("invalid shape for a and b");
    }

    let mut x = Array::zeros(a.shape()[1]);
    let mut p = Vec::new();
    p.resize_with(a.shape()[0], Default::default);

    let mut c_p = Default::default();
    lu(a, &mut p, &mut c_p)?;

    solve_lu_util(&a, &b, &p, &mut x)?;

    Ok(x)
}

pub fn det(a: &mut Array<f64, Ix2>, count_permute: usize) -> Option<f64> {
    if a.shape()[0] != a.shape()[1] { None }
    else {
        let mut v = 1.;
        for i in 0..a.shape()[0] {
            v *= a[[i,i]];
        }
        if (a.shape()[0] - count_permute) % 2 == 0 { Some(v) }
        else { Some(-v) }
    }
}

pub fn lu(a: &mut Array<f64, Ix2>, p: &mut Vec<usize>, count_permute: & mut usize) -> Result<(), &'static str> {
    
    let cols = a.shape()[1];
    let rows = a.shape()[0];
    
    //permutation of row
    p.resize_with(rows, Default::default);
    for i in 0..rows {
        p[i] = i;
    }
    *count_permute = 0;
    
    for i in 0..rows {
        //partial pivot
        let (mut v, mut ii) = (a[[p[i], i]], p[i]);
        for j in i..rows {
            let w = a[[p[j], i]];
            if w.abs() > v.abs() {
                v = w;
                ii = p[j];
            }
        }

        if v.abs() < Float::epsilon() {
            return Err("singular matrix");
        }

        p[i] = ii; //permute
        p[ii] = i;
        *count_permute += 1;
        
        for j in i + 1..rows {
            let jj = p[j];
            let f = a[[jj, i]] / a[[ii, i]];
            a[[jj, i]] = f;
            for k in i + 1..cols {
                a[[jj, k]] = a[[jj, k]] - f * a[[ii, k]];
            }
        }
    }

    Ok(())
}

pub fn solve_lu_util(
    a: &Array<f64, Ix2>,
    b: &Array<f64, Ix1>,
    p: &[usize],
    x: &mut Array<f64, Ix1>,
) -> Result<(), &'static str> {
    use std::cmp;

    let cols = a.shape()[1];
    let rows = a.shape()[0];

    //Ly = b
    //forward
    for i in 0..cmp::min(rows, cols) {
        let mut bb = b[p[i]];
        for j in 0..i {
            bb -= a[[i, j]] * x[j];
        }
        x[i] = bb;
    }

    if rows > cols {
        //overconstrained
        return Err("overconstrained");
    } else if rows < cols {
        //underconstrained
        for i in rows..cols {
            x[i] = 1.; //set to arbitary values
        }
    }

    //Ux = y
    //backward
    for i in (0..cmp::min(rows, cols)).rev() {
        let mut bb = x[i];
        for j in i + 1..cols {
            bb -= a[[i, j]] * x[j];
        }
        x[i] = bb / a[[i, i]];
    }

    Ok(())
}

fn permute(a: &mut Array<f64, Ix2>, p: &mut [usize]) {
    for idx in 0..p.len() {
        let i = p[idx];
        if idx != i {
            for j in 0..a.shape()[1] {
                a.swap([idx, j], [i, j]);
                p[idx] = idx;
                p[i] = i;
            }
            // let mut it = a.axis_iter_mut(Axis(0));
            // ndarray::Zip::from(it.nth(idx).unwrap())
            //     .and(it.nth(i).unwrap())
            //     .apply(std::mem::swap);
            // p[idx] = idx;
            // p[i] = i;
        }
    }
}

#[test]
fn test_lu_0() {
    let mut a = arr2(&[[4., 3.], [6., 3.]]);

    let orig = a.clone();

    let mut p = Vec::new();
    let mut c_p = 0;
    lu(&mut a, &mut p, & mut c_p).expect("lu");
    
    let mut pp = p.clone();
    permute(&mut a, &mut pp[..]);
    
    let mut l = a.clone();
    l[[0, 0]] = 1.;
    l[[1, 1]] = 1.;
    l[[0, 1]] = 0.;

    let mut u = a.clone();
    u[[1, 0]] = 0.;

    let mut r = l.dot(&u);
    permute(&mut r, &mut p[..]);

    assert!(arr2_eq(&orig, &r));
}

#[test]
fn test_lu_1() {
    let mut a = arr2(&[[6., 18., 3.], [2., 12., 1.], [4., 15., 3.]]);

    let orig = a.clone();

    let mut p = Vec::new();
    let mut c_p = 0;
    lu(&mut a, &mut p, &mut c_p).expect("lu");

    let mut l = a.clone();
    l[[0, 0]] = 1.;
    l[[1, 1]] = 1.;
    l[[2, 2]] = 1.;
    l[[0, 1]] = 0.;
    l[[1, 2]] = 0.;
    l[[0, 2]] = 0.;

    let mut u = a.clone();
    u[[1, 0]] = 0.;
    u[[2, 0]] = 0.;
    u[[2, 1]] = 0.;

    let mut r = l.dot(&u);
    permute(&mut r, &mut p[..]);

    assert!(arr2_eq(&orig, &r));
}

#[test]
fn test_solve_lu_util_0() {
    let mut a = arr2(&[[6., 18., 3.], [2., 12., 1.], [4., 15., 3.]]);

    let orig = a.clone();

    let b = arr1(&[3., 19., 0.]);

    let mut p = Vec::new();
    let mut c_p = 0;
    lu(&mut a, &mut p, &mut c_p).expect("lu");

    let mut x = Array::zeros(3);
    solve_lu_util(&a, &b, &p, &mut x).expect("solve_lu_util");

    let r = orig.dot(&x);

    assert!(arr1_eq(&r, &arr1(&[3., 19., 0.])));
}

#[test]
fn test_solve_lu_util_singular() {
    let mut a = arr2(&[[2., 12., 1.], [6., 18., 3.], [2., 12., 1.]]);

    let mut p = Vec::new();
    let mut c_p = Default::default();
    let err = lu(&mut a, &mut p, &mut c_p).expect_err("singular result expected");

    dbg!(err);
}

#[test]
fn test_solve_lu() {
    let mut a = arr2(&[[6., 18., 3.], [2., 12., 1.], [4., 15., 3.]]);

    let orig = a.clone();

    let b = arr1(&[3., 19., 0.]);

    let x = solve_lu(&mut a, &b).expect("solve_lu");

    let r = orig.dot(&x);

    assert!(arr1_eq(&r, &arr1(&[3., 19., 0.])));
}

#[test]
fn test_solve_lu_underconstrained() {
    let mut a = arr2(&[[1., 1., 1.]]);

    let orig = a.clone();

    let b = arr1(&[10.]);

    let x = solve_lu(&mut a, &b).expect("solve_lu");

    let r = orig.dot(&x);

    assert!(arr1_eq(&r, &arr1(&[10.])));
}
