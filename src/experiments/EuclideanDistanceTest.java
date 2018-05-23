package experiments;

import representation.xy.XYFilter;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.fileIO.DataSets;
import weka.classifiers.lazy.kNN;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import experiments.XY.DistanceType;

public class EuclideanDistanceTest {
	public static void main(String[] args) throws Exception {
		String[] datasets = DataSets.ucrNames;
		// String[] datasets = {"GunPoint","Coffee"};
		String dataDir = "G:/Êý¾Ý/TSC Problems/";
		Instances train, test, dTrain, dTest;
		EuclideanDistance ed;
		kNN knn;
		int correct;
		double acc, err;

		// important - use the correct one! Gorecki uses different derivatives
		// to Keogh
		XYFilter derFilter = new XYFilter();

		StringBuilder st = new StringBuilder();
		System.out
				.println("Dataset \t ED   \t XY_ED \t XY-ED-fixed");

		for (String dataset : datasets) {
			System.out.print(dataset + " \t ");

			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");

			dTrain = derFilter.process(train);
			dTest = derFilter.process(test);

			// ED
			ed = new EuclideanDistance();
			knn = new kNN(ed);
			correct = getCorrect(knn, dTrain, dTest);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			// XY_ED
			XY xy_ed = new XY(DistanceType.EUCLIDEAN);
			correct = getCorrect(xy_ed, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");
			
			// XY-ED-Fixed
			XY xy_ed_f = new XY(DistanceType.EUCLIDEAN);
			xy_ed_f.setAandB(0.5, 0.5);
			correct = getCorrect(xy_ed_f, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			System.out.println();
		}
	}

	protected static int getCorrect(kNN knn, Instances train, Instances test)
			throws Exception {
		knn.buildClassifier(train);
		int correct = 0;
		for (int i = 0; i < test.numInstances(); i++) {
			if (test.instance(i).classValue() == knn.classifyInstance(test
					.instance(i))) {
				correct++;
			}
		}
		return correct;
	}
}
