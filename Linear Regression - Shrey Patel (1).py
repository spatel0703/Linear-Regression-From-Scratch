#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#this generates a random array of 100 numbers, the random seed is so that I am returned the same values everytime
#this is to ensure consistency throughout the models
np.random.seed(42)
x = np.random.rand(100)
print(x)


# In[2]:


#this calculates the original y values by using gaussian noise
y = x + np.random.normal(0, 0.1, size=len(x))
print(y)


# In[3]:


#this is a scatterplot to show the relationship between x and y
plt.scatter(x,y)
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[4]:


#this is just a dataframe that combines both the x and y arrays into a concise dataset
data = pd.DataFrame({'x': x, 'y': y})
data


# In[5]:


#I saved the dataframe to a csv file for further use in Weka
data.to_csv('lin_regdata.csv', index=False)


# In[121]:


from sklearn.model_selection import train_test_split

#here I split the dataset into a training, validation, and test sets
train_split = 0.7
valid_split = 0.1
test_split = 0.2

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1 - train_split)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = test_split/(test_split+valid_split))

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


# In[122]:


#dataframe for training set
train_data = pd.DataFrame({'x': x_train, 'y': y_train})
train_data


# In[123]:


#dataframe for validation set
valid_data = pd.DataFrame({'x': x_valid, 'y': y_valid})
valid_data


# In[124]:


#dataframe for test set
test_data = pd.DataFrame({'x': x_test, 'y': y_test})
test_data


# In[125]:


#since we already have our values for x and y, for all sets, we do not need to calculate for that
#however, we still need to find m(slope) and b(intercept). Thus, I will be using the least squares calculation
def lin_regress(instance, output):
    input_mean = np.mean(instance)
    output_mean = np.mean(output)
    least_squares_num = (instance - input_mean) * (output-output_mean)
    least_squares_den = (instance - input_mean) ** 2
    m = least_squares_num / least_squares_den
    b = output - (m*input_mean)
    m, b = np.polyfit(instance, output, 1)
    return m, b

#this function will use the instances(x) from the dataset, then calculate the prediction(y) based on the calculated m and b
def predictor_function(instance, m, b):
    return (m * instance) + b

m, b = lin_regress(x_train, y_train)

print('m is:', m)
print('b is:', b)


# In[126]:


#this is caluclations for the validation set predictions
#this helps in testing the accuracy of the model
validation_predictions = predictor_function(x_valid, m, b)
validation_predictions


# In[127]:


#this prints out the actual values and compares it to the predictions using the validation set
print("Actual vs Validations Predictions:")
for actual, predicted in zip(y_test, validation_predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")


# In[128]:


#these functions calculate the errors and residuals between the actual and prediction values
#this helps show how accurate the model is
def mean_squared_error(actual, prediction):
    mse = ((actual - prediction)**2) / len(actual)
    return mse
def mean_absolute_error(actual, prediction):
    mae = (actual - prediction) / len(actual)
    return abs(mae)
def r_squared(actual, prediction):
    actual_mean = sum(actual) / len(actual)
    total_sum_squares = sum((actual - actual_mean) ** 2)
    residual_sum_squares = sum((actual - prediction) ** 2)
    r_2 = 1 - (residual_sum_squares / total_sum_squares)
    return r_2
def root_mean_squared_error(meansquareerror):
    rmse = np.sqrt(meansquareerror)
    return rmse


# In[129]:


#these are the errors and residuals of the validation predictions
mse_valid = mean_squared_error(y_valid, validation_predictions).sum()
mae_valid = mean_absolute_error(y_valid, validation_predictions).sum()
r2_valid = r_squared(y_valid, validation_predictions).sum()
rmse_valid = root_mean_squared_error(mse_valid)

print("Mean Squared Error:", mse_valid)
print("Mean Absolute Error:", mae_valid)
print("R-squared:", r2_valid)
print('Root Mean Squared Error:', rmse_valid)


# In[130]:


#this calculates the predictions for the test set
test_predictions = predictor_function(x_test, m, b)

print(test_predictions)


# In[131]:


#prints actual vs prediction for test predictions
print("Actual vs Predicted:")
for actual, predicted in zip(y_test, test_predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")


# In[132]:


#these are the errors and residuals of the test set predictions
mse_test = mean_squared_error(y_test, test_predictions).sum()
mae_test = mean_absolute_error(y_test, test_predictions).sum()
r2_test = r_squared(y_test, test_predictions).sum()
rmse_test = root_mean_squared_error(mse_test)

print("Mean Squared Error:", mse_test)
print("Mean Absolute Error:", mae_test)
print("R-squared:", r2_test)
print('Root Mean Squared Error:', rmse_test)


# In[133]:


#this plots the original data points vs the new regression line with the test set predictions

plt.scatter(x, y, color='blue', label='Original Data points')
plt.plot(x_test, test_predictions, color='red', label='Regression Line')
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')
plt.title('Linear Regression Model')
plt.legend()
plt.show()


# In[134]:


#here, I created a completely new linear regression and prediction function to handle iterations, and gradient descent
#the one above did not use iterations
def newlin_regress(instance, output, iterations, learning_rate=0.01):
    #here, I initalized the slope and intercept to 0
    #also made list to store gradient values after each iteration
    gradients_m = []  
    gradients_b = []
    
    nm = 0
    nb = 0
    
    #to go through the iterations, I made a for loop
    #each time an iteration is going through the loop, updates are made to the parameters
    for _ in range(iterations):
        predictions = (nm * instance) + nb
        error = predictions - output
        
        #updates the parameters based on the mean squared error derivative
        dm = (2/len(instance)) * np.sum(error * instance)
        db = (2/len(instance)) * np.sum(error)
        
        #updates m, b, and also gradient lists for m and b
        
        nm -= learning_rate * dm
        nb -= learning_rate * db
        gradients_m.append(dm)
        gradients_b.append(db)
        print(nm, nb)
    
    return nm, nb, gradients_m, gradients_b

def newpredictor_function(instance, nm, nb):
    return (nm * instance) + nb


nm, nb, gm, gb = newlin_regress(x_train, y_train, 10000)
print('m is:', nm)
print('b is:', nb)


# In[135]:


#this will show plot for 100 iterations, thus 100 gradient values for each m and b
#later in the code, I go through several different iteration numbers. this
#from the returned values in the last cell, you can see the steady growth of the gradient values as each iteration goes by
plt.plot(range(10000), gm, label='Slope Gradient') 
plt.plot(range(10000), gb, label='Intercept Gradient')
plt.xlabel("Iterations")
plt.ylabel("Gradient Values") 
plt.title("Gradient Descent Convergence")
plt.legend()
plt.show()


# In[136]:


#this will give the new validation predictions with gradient descent function
newvalidation_predictions = newpredictor_function(x_valid, nm, nb)
newvalidation_predictions


# In[137]:


#prints out actual vs new validation predictions 
print("Actual vs Predicted:")
for actual, predicted in zip(y_test, newvalidation_predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")


# In[138]:


#error and residual scores for new validation predictions
new_mse_valid = mean_squared_error(y_valid, newvalidation_predictions).sum()
new_mae_valid = mean_absolute_error(y_valid, newvalidation_predictions).sum()
new_r2_valid = r_squared(y_valid, newvalidation_predictions).sum()
new_rmse_valid = root_mean_squared_error(new_mse_valid)

print("Mean Squared Error:", new_mse_valid)
print("Mean Absolute Error:", new_mae_valid)
print("R-squared:", new_r2_valid)
print('Root Mean Squared Error:', new_rmse_valid)


# In[139]:


#this will give the new test set predictions with gradient descent function
newtest_predictions = newpredictor_function(x_test, nm, nb)

print(newtest_predictions)


# In[140]:


#prints out actual vs new test set predictions 
print("Actual vs Predicted:")
for actual, predicted in zip(y_test, newtest_predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")


# In[141]:


#error and residual scores for new test set predictions
newmse_test = mean_squared_error(y_test, newtest_predictions).sum()
newmae_test = mean_absolute_error(y_test, newtest_predictions).sum()
newr2_test = r_squared(y_test, newtest_predictions).sum()
newrmse_test = root_mean_squared_error(newmse_test)

print("Mean Squared Error:", newmse_test)
print("Mean Absolute Error:", newmae_test)
print("R-squared:", newr2_test)
print('Root Mean Squared Error:', newrmse_test)


# In[142]:


#plots original data points against regression line of the iterated test set predictions
plt.scatter(x, y, color='blue', label='Original Data points')
plt.plot(x_test, newtest_predictions, color='red', label='Regression Line')
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')
plt.title('Iterated Linear Regression Model')
plt.legend()
plt.show()


# In[143]:


iterations = [50,100,1600]
gradients_m = []
gradients_b = []

for itr in iterations:

    #will calculate gradient values from the gradient descent linear regression function
    nm, nb, gm, gb = newlin_regress(x_train, y_train, itr)
    
    #stores all of the gradient values, since original list will be empty after running through regression function each time
    gradients_m.append(gm[-1]) 
    gradients_b.append(gb[-1])


# In[144]:


#this will plot the iterations vs the gradient values for the slope and intercept 
plt.plot(iterations, gradients_m, label='Slope')
plt.plot(iterations, gradients_b, label='Intercept')
plt.xlabel("Iterations")
plt.ylabel("Gradient Values") 
plt.title("Gradient Descent Convergence")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




