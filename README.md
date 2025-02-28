========================================================================================================

                     **Instruction manual for the bachelor's thesis of Zhaoyu Wu**

========================================================================================================

Index
|
|
|---Brief description of each of the main files
|   |
|   |---Files necessary for the execution of the program
|   |
|   |---Test files(directly executable)
|   
|---Component diagramm
|
|---Procedure for starting the program




                              *Brief description of each of the main files*

--------------------------------------------------------------------------------------------------------
     Files necessary for the execution of the program
--------------------------------------------------------------------------------------------------------

src\ENV.py : Environment to generate the Instance for Model training

src\DDPG.py : Machine learning model DDPG

src\App.py : Backend startup file

src\App.js : Frontend startup file

src\Path_generator.py : Generating file for frontend graphics

src\Preprocess.py : Preprocessing file for received frontend data

src\DDPG_apply.py : In this file, DDPG is applied to user data.

--------------------------------------------------------------------------------------------------------
     Test files(directly executable)
--------------------------------------------------------------------------------------------------------

src\Test of Environment\step_function_test.py : Quick test for step function of environment

src\Test of Environment\reward_function_test.py : Quick test for reward function of Environment



src\Loss curve\Loss_Actor.py : Visual representation plot of the change in Actor's loss values for different learning rates

src\Loss curve\Loss_Actor_average.py : Visual representation plot of the average Actor's loss values for different learning rates

src\Loss curve\Loss_Actor_Changerate.py : Visual representation plot of the derivative of the Actor's loss function for different learning rates



src\Learningrate 3D\Loss_Actor_3d : When estimating the optimal learning rate for Actor and Critic, it needs to be applied to the 3D view.



                              *Component diagramm*
--------------------------------------------------------------------------------------------------------


+-------------------------------------------------------+
|                                                       |
|  +------------------+                                 |
|  |  <<Python>>      |                                 |
|  |  Environment     |                                 |
|  +------------------+                                 |
|           |                                           |
|           |                                           |
|  +------------------+        +-------------------+    |
|  |  <<Python>>      |        |  <<SQLite>>       |    |
|  |  Trainer         |--------|  Database         |    |
|  +------------------+        +-------------------+    |
|           |                            |              |
|           |                            |              |
|  +------------------+        +-------------------+    |
|  |  <<HDF5>>        |        |  <<Flask>>        |    |
|  |  DDPG            |--------|  Data Transmitter |----|---+
|  +------------------+        +-------------------+    |   |
|           |                            |   |          |   |
|           |                            |   |          |   |
|  +------------------+                  |   |          |   |
|  |  <<Python>>      |                  |   |          |   |
|  |  Data Processor  |------------------+   |          |   |
|  +------------------+                      |          |   |
|                                            |          |   |
|                                            |          |   |
|  +------------------+                      |          |   |
|  |  <<Python>>      |                      |          |   |
|  |  Path Generator  |----------------------+          |   |
|  +------------------+                                 |   |
|                                                       |   |
+-------------------------------------------------------+   |
                                                            |
                                                            |
+------------------------------------------------------------+
|  <<Web Application>>                                        |
|  User Interface                                             |
+------------------------------------------------------------+

--------------------------------------------------------------------------------------------------------



                              *Procedure for starting the program*
--------------------------------------------------------------------------------------------------------

1. run src\App.py

2. Start the foreground file by typing "npm start" at terminal in the directory where src\App.js is located.
