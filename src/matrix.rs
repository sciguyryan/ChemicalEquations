use std::ops::{Add, AddAssign};

#[derive(Debug)]
pub struct Matrix {
    m: Vec<Vec<f32>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0);

        let mut matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            matrix.push(vec![0.0; cols]);
        }

        Self { m: matrix }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.row_count() && col < self.column_count());

        self.m[row][col]
    }

    pub fn get_row(&self, row: usize) -> Vec<f32> {
        assert!(row < self.row_count());

        self.m[row].clone()
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        assert!(row < self.row_count() && col < self.column_count());

        self.m[row][col] = val;
    }

    pub fn set_row(&mut self, index: usize, row: Vec<f32>) {
        assert!(index < self.row_count());
        assert!(row.len() == self.column_count());

        self.m[index] = row;
    }

    pub fn swap_rows(&mut self, index_1: usize, index_2: usize) {
        assert_ne!(index_1, index_2);
        assert!(index_1 < self.row_count() && index_2 < self.row_count());

        self.m.swap(index_1, index_2);
    }

    pub fn multiply_row(&mut self, row_index: usize, scalar: f32) {
        assert!(row_index < self.row_count());

        for entry in &mut self.m[row_index] {
            *entry *= scalar;
        }
    }

    pub fn multiply_all(&mut self, scalar: f32) {
        for row in &mut self.m {
            for entry in row {
                *entry *= scalar;
            }
        }
    }

    pub fn gcd_row(&self, row_index: usize) -> f32 {
        assert!(row_index < self.row_count());

        let mut result = 0.0;
        for entry in &self.m[row_index] {
            result = Matrix::gcd(*entry, result);
        }

        result
    }

    fn gcd(mut a: f32, mut b: f32) -> f32 {
        a = a.abs();
        b = b.abs();

        while b != 0.0 {
            if b < a {
                std::mem::swap(&mut b, &mut a);
            }
            b %= a;
        }

        a
    }

    fn simplify_row(&self, row_index: usize) -> Vec<f32> {
        assert!(row_index < self.row_count());

        // Calculate the GCD of the row.
        let gdc = self.gcd_row(row_index);
        assert_ne!(gdc, 0.0);

        let mut result = Vec::with_capacity(self.row_count());
        for entry in &self.m[row_index] {
            result.push(entry / gdc);
        }

        result
    }

    fn simplify_row_in_place(&mut self, row_index: usize) {
        assert!(row_index < self.row_count());

        // Calculate the GCD of the row.
        let gdc = self.gcd_row(row_index);
        assert_ne!(gdc, 0.0);

        for entry in &mut self.m[row_index] {
            *entry /= gdc;
        }
    }

    pub fn row_count(&self) -> usize {
        self.m.len()
    }

    pub fn column_count(&self) -> usize {
        self.m[0].len()
    }

    pub fn rows(&self) -> impl Iterator<Item = &Vec<f32>> + '_ {
        self.m.iter()
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.row_count() != other.row_count() || self.column_count() != other.column_count() {
            return false;
        }

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                if self.get(i, j) != other.get(i, j) {
                    return false;
                }
            }
        }

        true
    }
}
impl Eq for Matrix {}

impl Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Then adding a matrix, the dimensions of both matrices must be equal.
        assert!(self.column_count() == rhs.column_count() && self.row_count() == rhs.row_count());

        let r = self.row_count();
        let c = self.column_count();
        let mut matrix = Matrix::new(r, c);

        for i in 0..r {
            for j in 0..c {
                matrix.set(i, j, self.get(i, j) + rhs.get(i, j));
            }
        }

        matrix
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        // Then adding a matrix, the dimensions of both matrices must be equal.
        assert!(self.column_count() == rhs.column_count() && self.row_count() == rhs.row_count());

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                self.m[i][j] += rhs.get(i, j)
            }
        }
    }
}

#[cfg(test)]
mod tests_matrix {
    use std::panic;

    use super::Matrix;

    #[test]
    fn test_create() {
        let matrix = Matrix::new(1, 3);
        assert_eq!(matrix.row_count(), 1);
        assert_eq!(matrix.column_count(), 3);

        // This one should fail as zero rows are not valid.
        let r = panic::catch_unwind(|| {
            let _ = Matrix::new(0, 3);
        });
        assert!(r.is_err());

        // This one should fail as zero columns are not valid.
        let r = panic::catch_unwind(|| {
            let _ = Matrix::new(1, 0);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_get() {
        let mut matrix = Matrix::new(1, 3);
        matrix.set_row(0, vec![1.0, 2.0, 3.0]);

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let _ = matrix.get(1, 0);
        });
        assert!(r.is_err());

        // This should fail as the row index is larger than the number of columns.
        let r = panic::catch_unwind(|| {
            let _ = matrix.get(0, 4);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_get_row() {
        let test_row = vec![1.0, 2.0, 3.0];

        let mut matrix = Matrix::new(1, 3);
        matrix.set_row(0, test_row.clone());
        assert_eq!(matrix.get_row(0), test_row);

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let _ = matrix.get_row(1);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_set() {
        let mut matrix = Matrix::new(1, 1);
        matrix.set(0, 0, 1.0);
        assert_eq!(matrix.get(0, 0), 1.0);

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            let _ = m.set(2, 0, 0.0);
        });
        assert!(r.is_err());

        // This should fail as the row index is larger than the number of columns.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            let _ = m.set(0, 2, 0.0);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_set_row() {
        let mut matrix = Matrix::new(1, 1);
        matrix.set_row(0, vec![1.0]);
        assert_eq!(matrix.get_row(0), vec![1.0]);

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            let _ = m.set_row(1, vec![1.0]);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_swap_rows() {
        let mut matrix = Matrix::new(2, 1);
        matrix.set_row(0, vec![1.0]);
        matrix.set_row(1, vec![2.0]);

        matrix.swap_rows(0, 1);

        let mut reference = Matrix::new(2, 1);
        reference.set_row(0, vec![2.0]);
        reference.set_row(1, vec![1.0]);

        assert_eq!(matrix, reference);

        // This should fail as we can't swap a row with itself.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            m.swap_rows(1, 1);
        });
        assert!(r.is_err());

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            m.swap_rows(2, 1);
        });
        assert!(r.is_err());

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            m.swap_rows(1, 2);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_multiply_rows() {
        let mut matrix = Matrix::new(2, 1);
        matrix.set_row(0, vec![1.0]);
        matrix.set_row(1, vec![2.0]);

        matrix.multiply_row(0, 2.0);

        let mut reference = Matrix::new(2, 1);
        reference.set_row(0, vec![2.0]);
        reference.set_row(1, vec![2.0]);

        assert_eq!(matrix, reference);

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            m.multiply_row(2, 1.0);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_multiply_all() {
        let mut matrix = Matrix::new(2, 1);
        matrix.set_row(0, vec![1.0]);
        matrix.set_row(1, vec![2.0]);

        matrix.multiply_all(2.0);

        let mut reference = Matrix::new(2, 1);
        reference.set_row(0, vec![2.0]);
        reference.set_row(1, vec![4.0]);

        assert_eq!(matrix, reference);
    }

    #[test]
    fn test_gcd_row() {
        let tests = [
            (vec![2.0, 4.0], 2.0),
            (vec![2.0, -4.0], 2.0),
            (vec![-2.0, 4.0], 2.0),
            (vec![2.0, 5.0], 1.0),
            (vec![0.0, 2.0], 2.0),
            (vec![4.2, 2.1], 2.1),
            (vec![0.0, 0.0], 0.0),
        ];

        let mut matrix = Matrix::new(1, 2);

        for t in tests {
            matrix.set_row(0, t.0);

            let gcd = matrix.gcd_row(0);
            assert_eq!(gcd, t.1);
        }

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let m = Matrix::new(1, 1);
            _ = m.gcd_row(2);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_simplify_row() {
        let tests = [
            (vec![2.0, 4.0], vec![1.0, 2.0]),
            (vec![2.0, -4.0], vec![1.0, -2.0]),
            (vec![-2.0, 4.0], vec![-1.0, 2.0]),
            (vec![2.0, 5.0], vec![2.0, 5.0]),
            (vec![0.0, 2.0], vec![0.0, 1.0]),
            (vec![4.2, 2.1], vec![2.0, 1.0]),
        ];

        let mut matrix = Matrix::new(1, 2);

        for t in tests {
            matrix.set_row(0, t.0);

            let simple = matrix.simplify_row(0);
            assert_eq!(simple, t.1);
        }

        // We cannot divide by a zero, we expect a panic in that case.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            m.set_row(0, vec![0.0]);
            _ = m.simplify_row(0);
        });
        assert!(r.is_err());

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let m = Matrix::new(1, 1);
            _ = m.simplify_row(2);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_simplify_row_in_place() {
        let tests = [
            (vec![2.0, 4.0], vec![1.0, 2.0]),
            (vec![2.0, -4.0], vec![1.0, -2.0]),
            (vec![-2.0, 4.0], vec![-1.0, 2.0]),
            (vec![2.0, 5.0], vec![2.0, 5.0]),
            (vec![0.0, 2.0], vec![0.0, 1.0]),
            (vec![4.2, 2.1], vec![2.0, 1.0]),
        ];

        let mut matrix = Matrix::new(1, 2);

        for t in tests {
            matrix.set_row(0, t.0);
            matrix.simplify_row_in_place(0);
            assert_eq!(matrix.get_row(0), t.1);
        }

        // We cannot divide by a zero, we expect a panic in that case.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            m.set_row(0, vec![0.0]);
            m.simplify_row_in_place(0);
        });
        assert!(r.is_err());

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let mut m = Matrix::new(1, 1);
            m.simplify_row_in_place(2);
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_add() {
        let tests = [
            (vec![2.0, 4.0], vec![1.0, 2.0], vec![3.0, 6.0]),
            (vec![-2.0, 4.0], vec![1.0, 2.0], vec![-1.0, 6.0]),
            (vec![-2.0, -4.0], vec![-1.0, -2.0], vec![-3.0, -6.0]),
        ];

        for t in tests.clone() {
            let mut matrix1 = Matrix::new(1, 2);
            matrix1.set_row(0, t.0);

            let mut matrix2 = Matrix::new(1, 2);
            matrix2.set_row(0, t.1);

            let result = matrix1 + matrix2;
            assert_eq!(result.get_row(0), t.2);
        }

        // This should fail as the matrices have different dimensions.
        let r = panic::catch_unwind(|| {
            let m1 = Matrix::new(1, 1);
            let m2 = Matrix::new(2, 1);
            _ = m1 + m2;
        });
        assert!(r.is_err());

        // Testing add assign.
        for t in tests {
            let mut matrix1 = Matrix::new(1, 2);
            matrix1.set_row(0, t.0);

            let mut matrix2 = Matrix::new(1, 2);
            matrix2.set_row(0, t.1);

            matrix1 += matrix2;
            assert_eq!(matrix1.get_row(0), t.2);
        }

        // This should fail as the matrices have different dimensions.
        let r = panic::catch_unwind(|| {
            let mut m1 = Matrix::new(1, 1);
            let m2 = Matrix::new(2, 1);
            m1 += m2;
        });
        assert!(r.is_err());
    }
}
