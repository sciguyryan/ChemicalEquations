use std::ops::{Add, AddAssign, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Debug)]
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

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        assert!(row < self.row_count() && col < self.column_count());

        self[row][col] = val;
    }

    pub fn set_row(&mut self, index: usize, row: Vec<f32>) {
        assert!(index < self.row_count());
        assert!(row.len() == self.column_count());

        self[index] = row;
    }

    pub fn add_rows(&mut self, index_1: usize, index_2: usize) -> Vec<f32> {
        assert_ne!(index_1, index_2);
        assert!(index_1 < self.row_count() && index_2 < self.row_count());

        let mut result = Vec::with_capacity(self.column_count());

        for i in 0..self.column_count() {
            result.push(self[index_1][i] + self[index_2][i]);
        }

        result
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

    pub fn gcd_row(&self, row_index: usize) -> f32 {
        assert!(row_index < self.row_count());

        let mut result = 0.0;
        for entry in &self.m[row_index] {
            result = Matrix::gcd(*entry, result);
        }

        result
    }

    fn gcd(mut a: f32, mut b: f32) -> f32 {
        if a == 0.0 {
            return b;
        }
        if b == 0.0 {
            return a;
        }

        a = a.abs();
        b = b.abs();

        while b != 0.0 {
            if b < a {
                std::mem::swap(&mut a, &mut b);
            }
            b %= a;
        }

        a
    }

    pub fn simplify_row(&self, row_index: usize) -> Vec<f32> {
        assert!(row_index < self.row_count());

        // Calculate the GCD of the row.
        let gdc = self.gcd_row(row_index);
        assert_ne!(gdc, 0.0);

        let mut result = Vec::with_capacity(self.row_count());
        for entry in &self[row_index] {
            result.push(entry / gdc);
        }

        result
    }

    pub fn simplify_row_in_place(&mut self, row_index: usize) {
        assert!(row_index < self.row_count());

        // Calculate the GCD of the row.
        let gcd = self.gcd_row(row_index);
        assert_ne!(gcd, 0.0);

        for entry in &mut self[row_index] {
            *entry /= gcd;
        }
    }

    pub fn transpose(&self) -> Self {
        // With transposition, the columns and rows are reversed.
        let rows = self.column_count();
        let cols = self.row_count();

        let mut matrix = Matrix::new(rows, cols);

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                matrix.set(j, i, self[i][j]);
            }
        }

        matrix
    }

    pub fn transpose_in_place(&mut self) {
        // With transposition, the columns and rows are reversed.
        let rows = self.column_count();
        let cols = self.row_count();

        let mut transposed = Matrix::new(rows, cols);

        for i in 0..self.row_count() {
            self.m.push(vec![]);
            for j in 0..self.column_count() {
                transposed.set(j, i, self[i][j]);
            }
        }

        *self = transposed;
    }

    // Partial credit goes to Brian Z (brianzhouzc) for his Java implementation
    // which can be found here:
    // https://github.com/brianzhouzc/Chemical-Equation-Balancer/blob/master/Source Code/src/com/upas/eqbalancer/Balancer.java#L615
    // It proved very helpful as a point of reference when I got stuck./
    pub fn gauss_jordan_eliminate(&mut self) -> Matrix {
        let mut matrix = self.clone();

        let rows = matrix.row_count();
        let cols = matrix.column_count();

        eprintln!("Starting matrix: {:?}", matrix);

        let mut c = 0;
        'outer: for r in 0..rows {
            if c >= cols {
                break;
            }

            // Find the pivot row.
            let mut i = r;
            while matrix[i][c] == 0.0 {
                i += 1;
                if i == rows {
                    i = r;
                    c += 1;
                    if c == cols {
                        break 'outer;
                    }
                }
            }

            // Switch the rows, but only if they have a different index.
            if i != r {
                matrix.swap_rows(i, r);
            }

            // Reduction phase.
            let scale = matrix[r][c];
            for j in 0..cols {
                matrix[r][j] /= scale;
            }

            // Elimination phrase.
            for i in 0..rows {
                if i != r {
                    let scale = matrix[i][c];
                    for j in 0..cols {
                        matrix[i][j] -= scale * matrix[r][j];
                    }
                }
            }

            c += 1;
        }

        eprintln!("Ending matrix: {:?}", matrix);

        matrix
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
                if self[i][j] != other[i][j] {
                    return false;
                }
            }
        }

        true
    }
}
impl Eq for Matrix {}

impl Mul<f32> for Matrix {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut matrix = self;

        for row in &mut matrix.m {
            for entry in row {
                *entry *= rhs;
            }
        }

        matrix
    }
}

impl MulAssign<f32> for Matrix {
    fn mul_assign(&mut self, rhs: f32) {
        for row in &mut self.m {
            for entry in row {
                *entry *= rhs;
            }
        }
    }
}

impl Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // When adding two matrices, the dimensions of both matrices must be equal.
        assert!(self.column_count() == rhs.column_count() && self.row_count() == rhs.row_count());

        let r = self.row_count();
        let c = self.column_count();
        let mut matrix = Matrix::new(r, c);

        for i in 0..r {
            for j in 0..c {
                matrix.set(i, j, self[i][j] + rhs[i][j]);
            }
        }

        matrix
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        // When adding two matrices, the dimensions of both matrices must be equal.
        assert!(self.column_count() == rhs.column_count() && self.row_count() == rhs.row_count());

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                self[i][j] += rhs[i][j]
            }
        }
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // When subtracting two matrices, the dimensions of both matrices must be equal.
        assert!(self.column_count() == rhs.column_count() && self.row_count() == rhs.row_count());

        let r = self.row_count();
        let c = self.column_count();
        let mut matrix = Matrix::new(r, c);

        for i in 0..r {
            for j in 0..c {
                matrix.set(i, j, self[i][j] - rhs[i][j]);
            }
        }

        matrix
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        // When subtracting two matrices, the dimensions of both matrices must be equal.
        assert!(self.column_count() == rhs.column_count() && self.row_count() == rhs.row_count());

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                self[i][j] -= rhs[i][j]
            }
        }
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f32>;

    fn index(&self, row_index: usize) -> &Self::Output {
        assert!(row_index < self.row_count());

        &self.m[row_index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, row_index: usize) -> &mut Self::Output {
        assert!(row_index < self.row_count());

        &mut self.m[row_index]
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
            let _ = matrix[1][0];
        });
        assert!(r.is_err());

        // This should fail as the row index is larger than the number of columns.
        let r = panic::catch_unwind(|| {
            let _ = matrix[0][4];
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_get_row() {
        let test_row = vec![1.0, 2.0, 3.0];

        let mut matrix = Matrix::new(1, 3);
        matrix.set_row(0, test_row.clone());
        assert_eq!(matrix[0], test_row);

        // This should fail as the row index is larger than the number of rows.
        let r = panic::catch_unwind(|| {
            let _ = matrix[1];
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_set() {
        let mut matrix = Matrix::new(1, 1);
        matrix.set(0, 0, 1.0);
        assert_eq!(matrix[0][0], 1.0);

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
        assert_eq!(matrix[0], vec![1.0]);

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
    fn test_multiply() {
        // Multiply by a scalar.
        let mut matrix = Matrix::new(2, 1);
        matrix.set_row(0, vec![1.0]);
        matrix.set_row(1, vec![2.0]);

        let m2 = matrix.clone() * 2.0;

        let mut reference = Matrix::new(2, 1);
        reference.set_row(0, vec![2.0]);
        reference.set_row(1, vec![4.0]);

        assert_eq!(m2, reference);

        // Multiply by a scalar, with assignment.
        matrix *= 2.0;
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
            assert_eq!(matrix[0], t.1);
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
            assert_eq!(result[0], t.2);
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
            assert_eq!(matrix1[0], t.2);
        }

        // This should fail as the matrices have different dimensions.
        let r = panic::catch_unwind(|| {
            let mut m1 = Matrix::new(1, 1);
            let m2 = Matrix::new(2, 1);
            m1 += m2;
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_subtract() {
        let tests = [
            (vec![2.0, 4.0], vec![1.0, 2.0], vec![1.0, 2.0]),
            (vec![-2.0, 4.0], vec![1.0, 2.0], vec![-3.0, 2.0]),
            (vec![-2.0, -4.0], vec![-1.0, -2.0], vec![-1.0, -2.0]),
        ];

        for t in tests.clone() {
            let mut matrix1 = Matrix::new(1, 2);
            matrix1.set_row(0, t.0);

            let mut matrix2 = Matrix::new(1, 2);
            matrix2.set_row(0, t.1);

            let result = matrix1 - matrix2;
            assert_eq!(result[0], t.2);
        }

        // This should fail as the matrices have different dimensions.
        let r = panic::catch_unwind(|| {
            let m1 = Matrix::new(1, 1);
            let m2 = Matrix::new(2, 1);
            _ = m1 - m2;
        });
        assert!(r.is_err());

        // Testing subtract assign.
        for t in tests {
            let mut matrix1 = Matrix::new(1, 2);
            matrix1.set_row(0, t.0);

            let mut matrix2 = Matrix::new(1, 2);
            matrix2.set_row(0, t.1);

            matrix1 -= matrix2;
            assert_eq!(matrix1[0], t.2);
        }

        // This should fail as the matrices have different dimensions.
        let r = panic::catch_unwind(|| {
            let mut m1 = Matrix::new(1, 1);
            let m2 = Matrix::new(2, 1);
            m1 -= m2;
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_transpose() {
        let tests = [
            (
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![1.0, 3.0], vec![2.0, 4.0]],
            ),
            (vec![vec![1.0], vec![2.0]], vec![vec![1.0, 2.0]]),
        ];

        for t in tests {
            let mut matrix = Matrix::new(t.0.len(), t.0[0].len());
            for (i, r) in t.0.iter().enumerate() {
                matrix.set_row(i, r.clone());
            }

            let mut reference = Matrix::new(t.1.len(), t.1[0].len());
            for (i, r) in t.1.iter().enumerate() {
                reference.set_row(i, r.clone());
            }

            let transposed = matrix.transpose();

            assert_eq!(transposed, reference);
        }
    }

    #[test]
    fn test_transpose_in_place() {
        let tests = [
            (
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![1.0, 3.0], vec![2.0, 4.0]],
            ),
            (vec![vec![1.0], vec![2.0]], vec![vec![1.0, 2.0]]),
        ];

        for t in tests {
            let mut matrix = Matrix::new(t.0.len(), t.0[0].len());
            for (i, r) in t.0.iter().enumerate() {
                matrix.set_row(i, r.clone());
            }

            let mut reference = Matrix::new(t.1.len(), t.1[0].len());
            for (i, r) in t.1.iter().enumerate() {
                reference.set_row(i, r.clone());
            }

            matrix.transpose_in_place();

            assert_eq!(matrix, reference);
        }
    }
}
