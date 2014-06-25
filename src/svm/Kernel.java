package svm;

public interface Kernel {
	abstract double value(double[] x1, double[] x2);
}
