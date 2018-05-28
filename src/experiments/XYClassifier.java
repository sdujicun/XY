package experiments;

import experiments.XY.DistanceType;
import representation.xy.XYFilter;
import tsc_algorithms.NN_CID;
import utilities.ClassifierTools;
import utilities.fileIO.DataSets;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.DerivativeFilter;

/**
 *
 * @author jc
 * 
 * 
 */

public class XYClassifier implements Classifier {
	Classifier t;
	Classifier td;
	Classifier tx;
	Classifier ty;

	public XYClassifier(Classifier t, Classifier td, Classifier tx,
			Classifier ty) {
		this.t = t;
		this.td = td;
		this.tx = tx;
		this.ty = ty;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException("Not supported yet.");
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		double a = t.classifyInstance(instance);

		XYFilter filter = new XYFilter();
		DerivativeFilter dfilter = new DerivativeFilter();
		Instances temp = new Instances(instance.dataset(), 0);
		Instances tempD, tempX, tempY;
		temp.add(instance);

		tempD = dfilter.process(temp);
		tempX = filter.processX(temp);
		tempY = filter.processY(temp);
		double b = tx.classifyInstance(tempX.get(0));
		double c = ty.classifyInstance(tempY.get(0));
		double d = td.classifyInstance(tempY.get(0));
		if ((a - b == 0) || (a - c == 0) || (a - d == 0)) {
			return a;
		}
		if (b - c == 0 || b - d == 0) {
			return b;
		}
		if (c - d == 0) {
			return c;
		}

		return a;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException("Not supported yet.");
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException("Not supported yet.");
	}
	
	public static void main(String[] args) throws Exception{
		
		
		
		String[] datasets = DataSets.ucrNames;
		String dataDir = "G:/Êý¾Ý/TSC Problems/";
		

		System.out.println("Dataset  \t CID  \t XY-CID");;


		for (String dataset : datasets) {
//		for(int i=20;i<30;i++){
//			String dataset=datasets[i];
			System.out.print(dataset + " \t ");

			Instances train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			Instances test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");
			XYFilter filter = new XYFilter();
			DerivativeFilter dfilter = new DerivativeFilter();
			
			Instances trainD, trainX, trainY;
			

			trainD = dfilter.process(train);
			trainX = filter.processX(train);
			trainY = filter.processY(train);
			
			EuclideanDistance ed = new EuclideanDistance();
			kNN knnT=new kNN(1);
			knnT.buildClassifier(train);
			knnT.setDistanceFunction(ed);
			kNN knnTD=new kNN(1);
			knnTD.setDistanceFunction(ed);
			knnTD.buildClassifier(trainD);
			kNN knnTX=new kNN(1);
			knnTX.setDistanceFunction(ed);
			knnTX.buildClassifier(trainX);
			kNN knnTY=new kNN(1);
			knnTY.setDistanceFunction(ed);
			knnTY.buildClassifier(train);
			
			XYClassifier xy=new XYClassifier(knnT,knnTD,knnTX,knnTY);
			
			double accuracy1 = utilities.ClassifierTools.accuracy(test, knnT);
			double accuracy2 = utilities.ClassifierTools.accuracy(test, xy);
			System.out.print(accuracy1 + " \t "+accuracy2 + " \t ");

			System.out.println();
		}

	}
	
		
	
	
}
