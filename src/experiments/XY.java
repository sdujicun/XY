package experiments;

import representation.xy.XYFilter;
import tsc_algorithms.DD_DTW;
import tsc_algorithms.DD_DTW.DistanceType;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.fileIO.DataSets;
import weka.classifiers.lazy.kNN;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.DTW_DistanceBasic;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author jc
 * 
 * 
 */

public class XY extends kNN {

	protected XYEuclideanDistance distanceFunction;
	protected boolean paramsSet;
	protected boolean sampleForCV = false;
	protected double prop;

	public enum DistanceType {
		EUCLIDEAN, DTW
	};

	// defaults to Euclidean distance
	public XY() {
		super();
		this.distanceFunction = new XYDTW();
		this.paramsSet = false;

	}

	public XY(DistanceType distType) {
		super();
		if (distType == DistanceType.EUCLIDEAN) {
			this.distanceFunction = new XYEuclideanDistance();
		} else {
			this.distanceFunction = new XYDTW();
		}
		this.paramsSet = false;
	}

	public void setAandB(double a, double b) {
		this.distanceFunction.a = a;
		this.distanceFunction.b = b;
		this.paramsSet = true;
	}

	@Override
	public void buildClassifier(Instances train) {
		if (!paramsSet) {
			this.distanceFunction.crossValidateForAandB(train);
			paramsSet = true;
		}
		this.setDistanceFunction(this.distanceFunction);
		super.buildClassifier(train);
	}

	public static class XYEuclideanDistance extends EuclideanDistance {

		protected double a;
		protected double b;
		public boolean sampleTrain = true; // Change back to default to false

		public XYEuclideanDistance() {
			this.a = 1;
			this.b = 0;
			// defaults to no derivative input
		}

		public XYEuclideanDistance(Instances train) {
			// this is what the paper suggests they use, but doesn't reproduce
			// results.
			// this.crossValidateForAlpha(train);

			// when cv'ing for a = 0:0.01:1 and b = 1:-0.01:0 results can be
			// reproduced though, so use that
			this.crossValidateForAandB(train);
		}

		public XYEuclideanDistance(double a, double b) {
			this.a = a;
			this.b = b;
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {
			// double dist = 0;
			double distX = 0;
			double distY = 0;

			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			int length = first.numAttributes() - classPenalty;
			double angle = 2 * Math.PI / length;
			for (int i = 0; i < length; i++) {
				double value1 = first.value(i);
				double value2 = second.value(i);

				double thisAngle = angle * i;
				// dist += Math.pow((value1 - value2),2) ;
				distX += Math.pow((value1 - value2) * Math.cos(thisAngle), 2);
				distY += Math.pow((value1 - value2) * Math.sin(thisAngle), 2);

			}
			return (a * Math.sqrt(distX) + b * Math.sqrt(distY));
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {
			// double dist = 0;
			double distX = 0;
			double distY = 0;

			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			int length = first.numAttributes() - classPenalty;
			double angle = 2 * Math.PI / length;
			for (int i = 0; i < length; i++) {
				double value1 = first.value(i);
				double value2 = second.value(i);

				double thisAngle = angle * i;
				// dist += Math.pow((value1 - value2),2) ;
				distX += Math.pow((value1 - value2) * Math.cos(thisAngle), 2);
				distY += Math.pow((value1 - value2) * Math.sin(thisAngle), 2);

			}
			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

		

		// changed to now return the predictions of the best alpha parameter
		public double[] crossValidateForAandB(Instances tr) {
			Instances train = tr;
			if (sampleTrain) {
				tr = InstanceTools.subSample(tr, tr.numInstances() / 10, 0);
			}

			double[] labels = new double[train.numInstances()];
			for (int i = 0; i < train.numInstances(); i++) {
				labels[i] = train.instance(i).classValue();
			}

			double[] a = new double[101];
			double[] b = new double[101];

			for (int alphaId = 0; alphaId <= 100; alphaId++) {
				a[alphaId] =  (100.0 - alphaId) / 100;
				b[alphaId] =  alphaId *1.0/ 100;
			}

			int n = train.numInstances();
			int k = a.length;
			int[] mistakes = new int[k];

			double[] D;
			double[] L;
			double[] d;
			double dist;
			double dDist;

			double[][] LforAll = new double[n][];

			double[] individualDistances;

			for (int i = 0; i < n; i++) {

				D = new double[k];
				L = new double[k];
				for (int j = 0; j < k; j++) {
					D[j] = Double.MAX_VALUE;
				}

				for (int j = 0; j < n; j++) {
					if (i == j) {
						continue;
					}

					individualDistances = this.getNonScaledDistances(
							train.instance(i), train.instance(j));
					dist = individualDistances[0];
					dDist = individualDistances[1];

					d = new double[k];

					for (int alphaId = 0; alphaId < k; alphaId++) {
						d[alphaId] = a[alphaId] * dist + b[alphaId] * dDist;
						if (d[alphaId] < D[alphaId]) {
							D[alphaId] = d[alphaId];
							L[alphaId] = labels[j];
						}
					}
				}

				for (int alphaId = 0; alphaId < k; alphaId++) {
					if (L[alphaId] != labels[i]) {
						mistakes[alphaId]++;
					}
				}
				LforAll[i] = L;
			}

			int bsfMistakes = Integer.MAX_VALUE;
			int bsfAlphaId = -1;
			for (int alpha = 0; alpha < k; alpha++) {
				if (mistakes[alpha] < bsfMistakes) {
					bsfMistakes = mistakes[alpha];
					bsfAlphaId = alpha;
				}
			}

			this.a = a[bsfAlphaId];
			this.b = b[bsfAlphaId];
			double[] bestAlphaPredictions = new double[train.numInstances()];
			for (int i = 0; i < bestAlphaPredictions.length; i++) {
				bestAlphaPredictions[i] = LforAll[i][bsfAlphaId];
			}
			// System.out.println("a:"+this.a+"b:"+this.b);
			return bestAlphaPredictions;
		}

		public double getA() {
			return a;
		}

		public double getB() {
			return b;
		}

	}

	public static class XYDTW extends XYEuclideanDistance {

		public XYDTW() {
			super();
		}

		public XYDTW(Instances train) {
			super(train);
		}

		public XYDTW(double a, double b) {
			super(a, b);
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double[] distances = getNonScaledDistances(first, second);
			return a * distances[0] + b * distances[1];
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {

			double distX = 0;
			double distY = 0;

			// DTW dtw = new DTW();
			DTW_DistanceBasic dtw = new DTW_DistanceBasic();
			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			XYFilter filter = new XYFilter();
			Instances tempX = new Instances(first.dataset(), 0);
			Instances tempY = new Instances(first.dataset(), 0);
			tempX.add(first);
			tempX.add(second);
			tempY.add(first);
			tempY.add(second);
			try {
				tempX = filter.processX(tempX);
				tempY = filter.processY(tempY);
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}

			distX = dtw.distance(tempX.get(0), tempX.get(1));
			distY = dtw.distance(tempY.get(0), tempY.get(1), Double.MAX_VALUE);

			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

	}

	

	public static void recreateResultsTable() throws Exception {
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
		System.out.println("Dataset \t ED  \t DD_ED \t XY_ED \t DTW \t DDTW \t DD_DTW \t XY-DTW");

//		for (String dataset : datasets) {
		//for (int i=34;i<datasets.length;i++) {
		for (int i=10;i<20;i++) {
			String dataset=datasets[i];
			System.out.print(dataset + " \t ");

			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");

			// instance resampling happens here, seed of 0 means that the
			// standard train/test split is used

			dTrain = derFilter.process(train);
			dTest = derFilter.process(test);



			//ED
//			ed = new EuclideanDistance();
//			knn = new kNN(ed);
//			correct = getCorrect(knn, dTrain, dTest);
//			acc = (double) correct / test.numInstances();
//			err = 1 - acc;
//			System.out.print(err + " \t ");

			// DD_ED
//			DD_DTW dd_ed = new DD_DTW(DD_DTW.DistanceType.EUCLIDEAN);
//			correct = getCorrect(dd_ed, train, test);
//			acc = (double) correct / test.numInstances();
//			err = 1 - acc;
//			System.out.print(err + " \t ");
			
			//XY_ED
			XY xy_ed = new XY(DistanceType.EUCLIDEAN);
			correct = getCorrect(xy_ed, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t "); 
			System.out.print(xy_ed.distanceFunction.a + " \t "); 
			System.out.print(xy_ed.distanceFunction.b + " \t "); 
			//XY-ED-Fixed
//			XY xy_ed_f = new XY(DistanceType.EUCLIDEAN);
//			xy_ed_f.setAandB(0.5, 0.5);
//			correct = getCorrect(xy_ed_f, train, test);
//			acc = (double) correct / test.numInstances();
//			err = 1 - acc;
//			System.out.print(err + " \t ");
			// DTW
			DTW_DistanceBasic dtw = new DTW_DistanceBasic();
			knn = new kNN(dtw);
			correct = getCorrect(knn, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			// DDTW
			DTW_DistanceBasic dDtw = new DTW_DistanceBasic();
			knn = new kNN(dDtw);
			correct = getCorrect(knn, dTrain, dTest);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			// DDDTW
			DD_DTW dd_dtw = new DD_DTW(DD_DTW.DistanceType.DTW);
			correct = getCorrect(dd_dtw, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			// XY_DTW
			XY xy_dtw = new XY(DistanceType.DTW);
			correct = getCorrect(xy_dtw, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err);
			System.out.print(xy_dtw.distanceFunction.a + " \t "); 
			System.out.print(xy_dtw.distanceFunction.b + " \t "); 
			
			//XY-DTW-Fixed
//			XY xy_dtw_f= new XY(DistanceType.DTW);
//			xy_dtw_f.setAandB(0.5, 0.5);
//			correct = getCorrect(xy_dtw_f, train, test);
//			acc = (double) correct / test.numInstances();
//			err = 1 - acc;
//			System.out.print(err + " \t ");
			
			System.out.println();
		}

	}

	public static void main(String[] args) {

		try {
			recreateResultsTable();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
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
