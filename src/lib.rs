extern crate ndarray;

#[cfg(test)]
mod util;

mod lu;
mod jacobi;
mod gs;
mod ssor;
mod cg;

pub mod prelude {
    pub use crate::jacobi::solve_jacobi as solve_jacobi;
    pub use crate::lu::solve_lu as solve_lu;
    pub use crate::gs::solve_gs as solve_gs;
    pub use crate::ssor::solve_ssor as solve_ssor;
    pub use crate::cg::solve_cg as solve_cg;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_crate() {
        use crate::prelude::*;
        use crate::ndarray::prelude::*;
        
        let mut a = arr2(&[[6., 18., 3.], [2., 12., 1.], [4., 15., 3.]]);

        let orig = a.clone();

        let b = arr1(&[3., 19., 0.]);

        let x = solve_lu(&mut a, &b).expect("solve_lu");

        let r = orig.dot(&x);

        assert!(crate::util::arr1_eq(&r, &arr1(&[3., 19., 0.])));
    }
}
