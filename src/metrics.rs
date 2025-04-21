#[derive(Debug)]
pub struct ConfusionMatrix(Box<[Box<[usize]>]>);

impl ConfusionMatrix {
    pub fn accuracy(&self) -> f64 {
        let total_correct: usize = self.0.iter().enumerate().map(|(i, row)| row[i]).sum();
        let total_samples: usize = self.0.iter().flatten().sum();
        total_correct as f64 / total_samples as f64
    }

    pub fn recall(&self, class: usize) -> f64 {
        let true_positive = self.0[class][class];
        let total_actual_positive: usize = self.0[class].iter().sum();
        true_positive as f64 / total_actual_positive as f64
    }
}

/// Compute the confusion matrix from a list of predicted and target labels.
pub fn confusion_matrix(
    predicted: &[usize],
    real: &[usize],
    num_classes: usize,
) -> ConfusionMatrix {
    assert_eq!(
        predicted.len(),
        real.len(),
        "Lengths of predicted and real labels must match"
    );

    let mut matrix =
        vec![vec![0usize; num_classes].into_boxed_slice(); num_classes].into_boxed_slice();

    for (&p, &r) in predicted.iter().zip(real.iter()) {
        matrix[r][p] += 1;
    }

    ConfusionMatrix(matrix)
}
