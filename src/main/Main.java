package main;

import java.util.Random;

import svm.Svm;

public class Main {
	public static void main(String[] args) {
		int l = 100;
		double[][] x = new double[l][2];
		double[] y = new double[l];
		
		Random rand = new Random(1);
		for (int i=0; i<l; i++) {
			x[i][0] = rand.nextDouble() * 2 - 1;
			x[i][1] = rand.nextDouble() * 2 - 1;
			
			if (x[i][0] * x[i][1] < 0) {
				y[i] = -1;
			} else {
				y[i] = 1;
			}
		}
		
		Svm svm = new Svm(x, y);
		
		svm.setC(10.0);
		
		long t1 = System.currentTimeMillis();
		svm.learn();
		System.out.println("learning time "+(System.currentTimeMillis() - t1)+"ms");
				
		int n = 1000;
		int error = 0;
		for (int i=0; i<n; i++) {
			double[] xx = new double[2];
			xx[0] = rand.nextDouble() * 2 - 1;
			xx[1] = rand.nextDouble() * 2 - 1;
			
			double yy = svm.get(xx);
			
			if (xx[0] * xx[1] < 0 && yy > 0 || xx[0] * xx[1] >= 0 && yy <= 0) {
				error += 1;
			}
		}
		System.out.println("error rate "+((double)error/n));
	}
}
