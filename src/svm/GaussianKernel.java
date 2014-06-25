package svm;

public class GaussianKernel implements Kernel {

	private double sigma = 1;
	
	public void setSigma(double s) {
		sigma = s;
	}
	
	@Override
	public double value(double[] x1, double[] x2) {
		double t = 0.0;
		for (int i=0; i<x1.length; i++) {
			t += Math.pow(x1[i] - x2[i], 2);
		}
		t /= (sigma*sigma);
		
		return Math.exp(-t);
	}

}
