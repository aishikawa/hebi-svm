package svm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Svm {
	private static double EPS = 1e-5; 
	
	private Kernel kernel;
	private double C = Double.MAX_VALUE / 2;
	
	private double[] alpha;
	
	private int l;
	private double[] target;
	private double[][] point;
	
	private Map<Integer, Double> error = new HashMap<Integer, Double>();
	
	public Svm(double[][] x, double[] y) {
		l = x.length;
		point = new double[l][];
		for (int i=0; i<l; i++) {
			point[i] = Arrays.copyOf(x[i], x[i].length);
		}
		target = Arrays.copyOf(y, y.length);
		
		kernel = new GaussianKernel();
	}
	
	public void setC(double c) {
		C = c;
	}
	
	public void learn() {
		alpha = new double[l];
		int numChanged = 0;
		boolean examineAll = true;
		while (numChanged > 0 || examineAll) {
			numChanged = 0;
			for (int i2=0; i2<l; i2++) {
				if (examineAll || (0 < alpha[i2] && alpha[i2] < C)) {
					numChanged += update(i2);
				}
			}
			if (examineAll) {
				examineAll = false;
			} else if (numChanged == 0) {
				examineAll = true;
			}
		}
	}
	
	private int update(int i2) {
		double y2 = target[i2];
		double alpha2 = alpha[i2];
		double E2;
		if (error.containsKey(i2)) {
			E2 = error.get(i2);
		} else {
			E2 = f(point[i2]) - y2;
			error.put(i2, E2);
		}
		double r2 = E2 * y2;
		if ((r2 < -EPS && alpha2 < C) || (r2 > EPS && alpha2 > 0)) {
			for (int i1=0; i1<l; i1++) {
				if (update(i1, i2, E2) > 0) {
					return 1;
				}
			}
		}
		return 0;
	}
	
	private int update(int i1, int i2, double E2) {
		if (i1 == i2) {
			return 0;
		}
		double alpha1 = alpha[i1];
		double alpha2 = alpha[i2];
		double y1 = target[i1];
		double y2 = target[i2];
		double E1;
		if (error.containsKey(i1)) {
			E1 = error.get(i1);
		} else {
			E1 = f(point[i1]) - y1;
			error.put(i1, E1);
		}

		double s = y1 * y2;
		double L, H;
		if (y1 != y2) {
			L = Math.max(0, alpha2 - alpha1);
			H = Math.min(C, C - alpha1 + alpha2);
		} else {
			L = Math.max(0,  alpha1 + alpha2 - C);
			H = Math.min(C, alpha1+alpha2);
		}
		if (H-L < EPS) {
			return 0;
		}
		double k11 = kernel.value(point[i1], point[i1]);
		double k12 = kernel.value(point[i1], point[i2]);
		double k22 = kernel.value(point[i2], point[i2]);
		double eta = 2 * k12 - k11 - k22;

		double a2 = alpha2 - y2 * (E1-E2) / eta;
		if (a2 < L) {
			a2 = L;
		} else if (a2 > H) {
			a2 = H;
		}

		if (Math.abs(a2 - alpha2) < EPS) {
			return 0;
		}
		double a1 = alpha1 + s * (alpha2 - a2);
		
		alpha[i1] = a1;
		alpha[i2] = a2;
				
		error.clear();
		return 1;
	}
	
	private double f(double[] x) {
		double ret = 0;
		for (int i=0; i<l; i++) {
			ret += alpha[i] * target[i] * kernel.value(point[i], x);
		}
		return ret;
	}
	
	public double get(double[] x) {
		double y = f(x);
		if (y < 0) {
			return -1.0;
		} else {
			return 1.0;
		}
	}
}
