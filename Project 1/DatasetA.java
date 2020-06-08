// Trang Do

import java.util.Random;
import java.lang.Math;


public class DatasetA{
	public static void main(String args[]){
		double totalRoll = 20;
		double numberOfOne = 0;
	    double numberOfTwo = 0;
	    double numberOfThree = 0;
	    double numberOfFour = 0;
	    double numberOfFive = 0;
	    double numberOfSix = 0;

	    // Generate random integers
		int[] randomNumbers = new int[] { 1,2,3,4,5,6,6,6,6,6 };
	    for (int i =0; i< totalRoll; i++){
	    	Random r = new Random();
	    	int nextRandomNumberIndex = r.nextInt(randomNumbers.length);
	    	int random = randomNumbers[nextRandomNumberIndex];
	    	System.out.println(random);
	    	// Calculate the number of each number appears
		    if (random == 1){
		    	numberOfOne ++;
		    }
		    if (random == 2){
		    	numberOfTwo ++;
		    }
		    if (random == 3){
		    	numberOfThree +=1;
		    }
		    if (random == 4){
		    	numberOfFour +=1;
		    }
		    if (random == 5){
		    	numberOfFive +=1;
		    }
		    if (random == 6){
		    	numberOfSix +=1;
		    }

		}

		// Scenario 1,  calculate MLE
		
	   	double mleOfOne = numberOfOne / totalRoll;
	    double mleOfTwo = numberOfTwo / totalRoll;
	    double mleOfThree = numberOfThree / totalRoll;
	    double mleOfFour = numberOfFour / totalRoll;
	    double mleOfFive = numberOfFive / totalRoll;
	    double mleOfSix = numberOfSix / totalRoll;

	    System.out.println("The probability of 1 using MLE is " + mleOfOne);
	    System.out.println("The probability of 2 using MLE is " + mleOfTwo);
	    System.out.println("The probability of 3 using MLE is " + mleOfThree);
	    System.out.println("The probability of 4 using MLE is " + mleOfFour);
	    System.out.println("The probability of 5 using MLE is " + mleOfFive);
	    System.out.println("The probability of 6 using MLE is " + mleOfSix);

	    // Scenario 2, probability of 6 is 1/6
	    double probOfSix = 1.0/6;
	    double probOfNotSix = 1 - probOfSix;
	    System.out.println("Bayesian inference in Scenario 2 is " + Math.pow(probOfSix, numberOfSix)*Math.pow(probOfNotSix, (totalRoll-numberOfSix)));

	    // Scenario 3, probability of 6 is 1/2
	    probOfSix = 1.0/2;
	    probOfNotSix = 1 - probOfSix;
	    System.out.println("Bayesian inference in Scenario 3 is " + Math.pow(probOfSix, numberOfSix)*Math.pow(probOfNotSix, (totalRoll-numberOfSix)));

	    // Scenerio 4, probability of 6 is 0.4
	    probOfSix = 0.4;
	    probOfNotSix = 1 - probOfSix;
	    System.out.println("Bayesian inference in Scenario 4 is " + Math.pow(probOfSix, numberOfSix)*Math.pow(probOfNotSix, (totalRoll-numberOfSix)));

	   

	}
}