# COMP390
Co-evolution With Nash-Memory

  Main:

    while iteration not end:
  
      Mutate Classifiers and Models (Using support set form last iteration)
    
      Calculate Fitness
    
      Sort by Fitness
    
      Find Best Response
    
      If best response is good:
    
        apply nash method to update WMN sets
      
      else:
    
        skip to next iteration
