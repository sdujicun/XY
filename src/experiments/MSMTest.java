package experiments;

import utilities.ClassifierTools;
import utilities.fileIO.DataSets;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.core.elastic_distance_measures.MSMDistance;
import experiments.XY.DistanceType;

public class MSMTest {
	public static void main(String[] args) throws Exception {
		String[] datasets = DataSets.ucrNames;
		String dataDir = "G:/Êý¾Ý/TSC Problems/";
		Instances train, test;
		MSMDistance msm;
		kNN knn;
		int correct;
		double acc, err;

		StringBuilder st = new StringBuilder();
		System.out
				.println("Dataset \t MSM   \t XY_MSM");

		for (String dataset : datasets) {
			System.out.print(dataset + " \t ");

			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");

			// MSM
			msm = new MSMDistance();
			knn = new kNN(msm);
			correct = getCorrect(knn, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			// XY_MSM
			XY xy_msm = new XY(DistanceType.MSM);
			xy_msm.setAandB(0.5, 0.5);
			correct = getCorrect(xy_msm, train, test);
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
