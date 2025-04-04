import numpy as np

def random_predict(number: int=1) -> int:
    
    count = 0
    
    while True:
        count += 1
        predict_number = np.random.randint(1, 101)
        if number == predict_number:
            break
        
    return count

def fast_predict(number: int=1) -> int:
    """ Algorithm that guesses a random number in O(log n) time
    by picking every consequetive guess from the
    middle of the range [min, max] determined by earlier attempts"""
    
    count = 0
    min, max = 0, 101
    
    while True:        
        count += 1
        predict_number = round((min+max) / 2)
        
        if number == predict_number:
            break
        elif number > predict_number:
            min = predict_number
        else:
            max = predict_number
            
    return count                

def score_game(predict_func) -> int:
    
    count_ls = []
    np.random.seed(1)
    random_array = np.random.randint(1, 101, size = (1000))
    for number in random_array:
        count_ls.append(predict_func(number))
        
    score = int(np.mean(count_ls))
    
    print(f"Your algorithm guesses numbers in an average of {score} attempts")
    return (score)

if __name__ == '__main__':
    score_game(random_predict)