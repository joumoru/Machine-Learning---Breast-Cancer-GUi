//package machinelearning1;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.awt.*;
import java.util.Objects;
import javax.swing.*;

public class BreastCancerClassifierGUI {
    private JFrame frame;
    private JComboBox[] attributeInputs;

    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            try {
                MachineLearning1.initialize();
                Instances datasetInstances = MachineLearning1.getDatasetInstances();
                BreastCancerClassifierGUI window = new BreastCancerClassifierGUI(datasetInstances);
                window.frame.setVisible(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    public BreastCancerClassifierGUI(Instances datasetInstances) {
        initialize();
    }

    private void initialize() {
        frame = new JFrame("Breast Cancer Classifier");
        frame.setBounds(100, 100, 450, 300);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().setLayout(new GridLayout(10, 2));

        String[][] attributeValues = {
                {"10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"},
                {"lt40", "ge40", "premeno"},
                {"0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"},
                {"0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"},
                {"yes", "no"},
                {"1", "2", "3"},
                {"left", "right"},
                {"left_up", "left_low", "right_up", "right_low", "central"},
                {"yes", "no"}
        };

        String[] attributeLabels = {
                "Age", "Menopause", "Tumor Size", "Inv Nodes", "Node Caps", "Deg Malig", "Breast", "Breast Quad", "Irradiat"
        };

        attributeInputs = new JComboBox[attributeValues.length];

        for (int i = 0; i < attributeValues.length; i++) {
            frame.getContentPane().add(new JLabel(attributeLabels[i]));
            attributeInputs[i] = new JComboBox<>(attributeValues[i]);
            frame.getContentPane().add(attributeInputs[i]);
        }

        JButton classifyButton = new JButton("Classify");
        classifyButton.addActionListener(e -> classify());
        frame.getContentPane().add(classifyButton);
    }

    private void classify() {
        try {
            // Get dataset instances from MachineLearning1 class
            Instances datasetInstances = MachineLearning1.getDatasetInstances();

            // Create a new instance with the user input
            Instance newInstance = new DenseInstance(datasetInstances.numAttributes());
            for (int i = 0; i < attributeInputs.length; i++) {
                newInstance.setValue(i, Objects.requireNonNull(attributeInputs[i].getSelectedItem()).toString());
            }

            // Set the class attribute for the new instance
            newInstance.setDataset(datasetInstances);

            // Add the new instance to the dataset instances
            datasetInstances.add(newInstance);

            // Classify the new instance
            NaiveBayes classifier = MachineLearning1.getClassifier();
            double result = classifier.classifyInstance(newInstance);

            // Get the class label for the result
            String classLabel = datasetInstances.classAttribute().value((int) result);

            // Show the result in a JOptionPane
            JOptionPane.showMessageDialog(frame, "The classification result is: " + classLabel);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(frame, "Error: " + e.getMessage());
        }
    }


}
