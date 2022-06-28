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
        assert!(index_1 < self.row_count() && index_2 < self.row_count());

        let tmp = self.m[index_1].clone();
        self.m[index_1] = self.m[index_2].clone();
        self.m[index_2] = tmp;
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

    fn gcd(mut n: f32, mut m: f32) -> f32 {
        assert!(n != 0.0 && m != 0.0);

        while m != 0.0 {
            if m < n {
                std::mem::swap(&mut m, &mut n);
            }
            m %= n;
        }

        n
    }

    fn simplify_row(&self, row_index: usize) -> Vec<f32> {
        assert!(row_index < self.row_count());

        // Calculate the GCD of the row.
        let gdc = self.gcd_row(row_index);

        let mut result = Vec::with_capacity(self.row_count());
        for entry in &self.m[row_index] {
            result.push(entry / gdc);
        }

        result
    }

    pub fn row_count(&self) -> usize {
        self.m.len()
    }

    pub fn column_count(&self) -> usize {
        self.m[0].len()
    }
}

#[cfg(test)]
mod tests_matrix {
    use std::panic;

    use super::Matrix;

    #[test]
    fn test_matrix_create() {
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
    fn test_matrix_get() {
        let mut matrix = Matrix::new(1, 3);
        matrix.set_row(0, vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix.get(0, 1), 2.0);

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
    fn test_matrix_get_row() {
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
    fn test_matrix_set() {
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
    fn test_matrix_set_row() {
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
}
