mod definitions;
pub mod formula_tools;
mod matrix;

use std::collections::HashMap;

use nalgebra::SMatrix;

use crate::{formula_tools::parser, matrix::Matrix};

fn main() {
    let els = vec!["Ca", "O", "H", "P"];
    let rterms: Vec<Vec<f32>> = Vec::new();
    let pterms: Vec<Vec<f32>> = Vec::new();

    let m = vec![
        vec![1.0, 2.0, 2.0, 0.0],
        vec![0.0, 4.0, 3.0, 1.0],
        vec![-3.0, -8.0, 0.0, -2.0],
        vec![0.0, -1.0, -2.0, 0.0],
    ];

    let mut matrix = Matrix::from(&m[..]);

    //eprintln!("{:?}", m);

    let rrrrr: Vec<f32> = m.into_iter().flatten().collect();
    eprintln!("{:?}", rrrrr);

    let mut test = SMatrix::<f32, 4, 4>::from_vec(rrrrr);

    eprintln!("{:?}", test);

    let mut transposed = test.transpose();

    let tttttttt = transposed.determinant();
    eprintln!("{:?}", tttttttt);

    return;

    //let mut m: Vec<Vec<f32>> = vec![m1, m2, m3, m4];
    //println!("{:#?}", m);

    // Solve the matrix into the RREF form.
    /*let mut m_solved = reduced_row_echelon_form(&m);
    println!("{:?}", m_solved);

     let mut coeffs = Vec::new();
     for i in 0..m_solved[0].len() {
         let len = m_solved[0].len() - 1;
         if m_solved[i][len] == 0.0 {
             m_solved[i][len] = 1.0;
         }
         coeffs.push(m_solved[i][len]);
     }
     println!("{:#?}", coeffs);

     // Calculate the multipliers.
     let mut elm = Vec::with_capacity(coeffs.len());
     let mut denoms =Vec::with_capacity(coeffs.len());
     for i in 0..coeffs.len() {
         elm.push(m_solved[i][coeffs.len() - 1]);

         let f = Fraction::from(elm[i]);
         let d = f.denom().expect("failed to get denominator");
         denoms.push(*d);
     }
     println!("{:#?}", elm);

     let factor = lcm_slice(&denoms);
     println!("{}", factor);

     let mut fin = Vec::with_capacity(elm.len());
     for e in &mut elm {
         *e *= factor as f32;
         fin.push(e.ceil());
     }
     println!("{:?}", fin);

    return;*/

    let formula = "(Zn2(Ca(BrO4))K(Pb)2Rb)33+";

    let cycles = 10000;

    use std::time::Instant;
    let now = Instant::now();
    {
        for _ in 0..cycles {
            let r = parser::parse::parse(formula);
            /*if let Ok(parsed) = r {
                //print_sorted(parsed);
                println!("{}", parsed.print());
            }*/
        }
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed / cycles);
}

fn print_sorted(map: HashMap<String, u32>) {
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));

    eprintln!("{:?}", v);
}
