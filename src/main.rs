mod definitions;
pub mod formula_tools;
mod matrix;

use std::collections::HashMap;

use crate::{formula_tools::parser, matrix::Matrix};

fn main() {
    let els = vec!["Ca", "O", "H", "P"];
    let rterms: Vec<Vec<f32>> = Vec::new();
    let pterms: Vec<Vec<f32>> = Vec::new();

    let m1 = vec![1.0, 2.0, 2.0, 0.0];
    let m2 = vec![0.0, 4.0, 3.0, 1.0];
    let m3 = vec![3.0, 8.0, 0.0, 2.0];
    let m4 = vec![0.0, 1.0, 2.0, 0.0];

    let mut m = Matrix::new(4, 4);
    m.set_row(0, m1);
    m.set_row(1, m2);
    m.set_row(2, m3);
    m.set_row(3, m4);

    //eprintln!("{:?}", m);

    let test = m.gauss_jordan_eliminate();
    //eprintln!("{:?}", m);

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
