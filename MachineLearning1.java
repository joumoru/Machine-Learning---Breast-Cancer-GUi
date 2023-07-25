import java.io.*;
import weka.classifiers.*;
import weka.classifiers.bayes.*;
import weka.core.*;

public class MachineLearning1 {
    private static NaiveBayes naivebayes;
    private static Instances datasetInstances;

    public static void main(String[] args) {
        try {
            initialize();
            // Rest of the main method code
        } catch (Exception e) {
            System.out.println("Error Occurred!!!! \n" + e.getMessage());
        }
    }

    public static void initialize() {
        try {
            naivebayes = new NaiveBayes();
            String dataset = "breast-cancer.arff";
            BufferedReader bufferedReader = new BufferedReader(new FileReader(dataset));

            // Create dataset instances //
            datasetInstances = new Instances(bufferedReader);

            // Randomize the dataset //
            datasetInstances.randomize(new java.util.Random(0));

            // Divide dataset into training and test data //
            int trainingDataSize = (int) Math.round(datasetInstances.numInstances() * 0.66);
            int testDataSize = (int) datasetInstances.numInstances() - trainingDataSize;

            // Create training data //
            Instances trainingInstances = new Instances(datasetInstances, 0, trainingDataSize);
            // Create test data //
            Instances testInstances = new Instances(datasetInstances, trainingDataSize, testDataSize);

            // Set Target class //
            trainingInstances.setClassIndex(trainingInstances.numAttributes() - 1);
            testInstances.setClassIndex(testInstances.numAttributes() - 1);

            // Close BufferedReader //
            bufferedReader.close();

            // Build Classifier //
            naivebayes.buildClassifier(trainingInstances);

            // Evaluation //
            Evaluation evaluation = new Evaluation(trainingInstances);
            evaluation.evaluateModel(naivebayes, testInstances);
            System.out.println(evaluation.toSummaryString("\nResults", false));
        } catch (Exception e) {
            System.out.println("Error Occurred!!!! \n" + e.getMessage());
        }
    }

    public static NaiveBayes getClassifier() {
        return naivebayes;
    }

    public static Instances getDatasetInstances() {
        return datasetInstances;
    }
}
