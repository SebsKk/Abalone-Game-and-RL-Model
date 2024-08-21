def sort_balls(balls_start, balls_end):
        if len(balls_start) > 1:
            set1 = set(balls_start)
            set2 = set(balls_end)
            common_elements = set1.intersection(set2)
            if len(common_elements) > 0:
                # Find the unique elements
                unique_start = list(set1 - set2)[0]
                unique_end = list(set2 - set1)[0]

                def distance_to_unique_end(ball):
                    return abs(ball[0] - unique_end[0]) + abs(ball[1] - unique_end[1])

                sorted_balls_start = sorted([ball for ball in balls_start if ball != unique_start], key=distance_to_unique_end)
                sorted_balls_end = sorted([ball for ball in balls_end if ball != unique_end], key=distance_to_unique_end)
            
                sorted_balls_start.append(unique_start)
                sorted_balls_end.insert(0, unique_end)
            
                return sorted_balls_start, sorted_balls_end
            else:
                if len(balls_start) == 2:
                    direction = (balls_start[0][0] - balls_start[1][0], balls_start[0][1] - balls_start[1][1])
                    if (balls_end[0][0] - balls_end[1][0], balls_end[0][1] - balls_end[1][1]) == direction:
                        return balls_start, balls_end 
                    else:
                        balls_end = balls_end[::-1]
                        return balls_start, balls_end
                else:
                    # For 3-ball parallel moves
                    sorted_balls_start = sorted(balls_start, key=lambda x: (x[0], x[1]))
                    
                    # Determine the direction of movement
                    direction = (balls_end[0][0] - balls_start[0][0], balls_end[0][1] - balls_start[0][1])
                    
                    # Sort balls_end in the same order as balls_start
                    sorted_balls_end = sorted(balls_end, key=lambda x: (x[0] - direction[0], x[1] - direction[1]))
                    
                    return sorted_balls_start, sorted_balls_end
        else:
            return balls_start, balls_end


    
        
print(sort_balls([(1, 4), (2, 4), (3, 4)], [(2, 4), (3, 4), (4, 4)]))  # Parallel move
