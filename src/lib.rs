extern crate ndarray;

#[cfg(test)]
mod util;

mod cg;
mod gs;
mod jacobi;
mod lu;
mod ssor;

pub mod prelude {
    pub use crate::cg::solve_cg;
    pub use crate::cg::solve_cg_precond;
    pub use crate::gs::solve_gs;
    pub use crate::jacobi::solve_jacobi;
    pub use crate::lu::solve_lu;
    pub use crate::ssor::solve_ssor;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_crate() {
        use crate::ndarray::prelude::*;
        use crate::prelude::*;

        let mut a = arr2(&[[6., 18., 3.], [2., 12., 1.], [4., 15., 3.]]);

        let orig = a.clone();

        let b = arr1(&[3., 19., 0.]);

        let x = solve_lu(&mut a, &b).expect("solve_lu");

        let r = orig.dot(&x);

        assert!(crate::util::arr1_eq(&r, &arr1(&[3., 19., 0.])));
    }
}
