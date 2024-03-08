# NN_airfoil_prediction

Background
How can machine learning approaches be used to support technical problems? 
And for which problems does it make sense to use them? In particular, use cases 
in the field of fluid mechanics are of my personal interest.

## GOAL
This example of a neural network (MLP) is intended to establish a relationship 
between airfoil contours with flow velocity and angle of attack as features 
of a set of labels consisting of aerodyanmic coefficients. Ideally, 
after training, it should be possible to evaluate an "arbitrary" 
airfoil geometry in such a way that one should obtain lift, drag and 
pitching moment coefficients.
Another goal of this framework here is to play around. It's purpose is not to 
deliver high sophisticated code but more to: 
1. Enable understanding of machine learning and handling of data
2. How do different algorithms and configurations influence the quality of the 
   prediction and learning speed
For this open heart surgery, the author prefers the KISS principle:
"Keep it simple, stupid!"


## DATASET
The original dataset is obtained from :
https://www.kaggle.com/datasets/swegmaster/airfoil-performance-and-geometry-data

Due to a problem with the reconstruction of the airfoil shape, the original data
have been modified. As already mentioned on the dataset link above, it is not
obvious how to reconstruct the shape from the coefficients. I mean, mathematically
it's clear how to do it. But first, it's not the best idea to parametrize an 
airfoil geometry  by a polynomial of order 30 as it becomes unstable and
second, the reconstructed shapes don't look like an airfoil.

That is why the coefficients in the dataset from Kaggle were replaced by a 
better parametrization with new coefficients. The method used, is the CST 
parametrization of Brenda Kulfan [AIAA 2006] with the nose modification [2020].
To build up the database of CST coefficients, another database with
airfoil coordinates was used (https://m-selig.ae.illinois.edu/ads/coord_database.html)

As a result, the data used, consist of the features:
airfoilName, kulfan_coeff_0, kulfan_coeff_1  .... kulfan_coeff_27, reynoldsNumber, alpha

Where:
-airfoilName is the airfoil name
-the 28 kulfan_coeff_[0-27] coefficients describe the upper and lower
airfoil geometry
-the Reynolds number is a dimensionless value for the flow velocity
-alpha is the angle of attack (AoA) in degree

The labels are:
coefficientLift, coefficientDrag, coefficientParasiteDrag, coefficientMoment, topXTR, botXTR

Where:
-coefficientLift is the dimensionless value for lift
-coefficientDrag is the dimensionless value for drag
-coefficientParasiteDrag is the dimensionless value for parasite drag, where 
 parasite means the combination of form drag and skin friction 
 (Important notice: In contrast to the official description on www.Kaggle.com,
  the author assumes that coefficientParasiteDrag means the pressure based 
  drag)
-coefficientMoment is the dimensionless moment coefficient
-topXTR is the dimensionless position of the transition from laminar to turbulent flow on the upper side
-botXTR is the dimensionless position of the transition from laminar to turbulent flow on the lower side

## MACHINE LEARNING APPROACH
The task to predict the aerodynamic performance of an airfoil by its shape description
is categorized as a regression problem. As a first concept a multilayer perceptron (MLP)
with the follwoing attributes is used:
- input layer with 30 neurons
- dense layer 100 neurons
- dropout layer
- dense layer 200 neurons
- dropout layer
- dense layer 100 neurons
- 6 output neurons
