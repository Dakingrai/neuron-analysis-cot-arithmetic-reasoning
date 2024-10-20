
cot_prompts ="""Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step
There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5.
The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let's think step by step
Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39.
The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step
Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8.
The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step
Shawn started with 5 toys.
If he got 2 toys each from his mom and dad, then that is 4 more toys.
5 + 4 = 9.
The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's think step by step
There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
9 + 20 is 29.
The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step
Michael started with 58 golf balls.
After losing 23 on tues- day, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = 33 golf balls.
The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step
Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left.
23 - 15 is 8.
The answer is 8.

"""

my_cot_prompts ="""Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step
First there are 15 trees.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6 trees.
The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step
First there are 3 cars.
Then 2 more cars arrive.
So there must be 3 + 2 = 5 cars.
The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let's think step by step
First Leah had 32 chocolates and her sister had 42 chocolates.
So in total they had 32 + 42 = 74 chocolates.
Then they ate 35 chocolates.
So there must be 74 - 35 = 39 chocolates.
The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step
First Jason had 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8 lollipops.
The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step
First Shawn has 5 toys.
Then he got 2 toys each from his mom and dad.
So he must have 5 + 4 = 9 toys.
The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's think step by step
First there were 9 computers.
Then for each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
So there must be in total 9 + 20 = 29 computers.
The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step
First Michael started with 58 golf balls.
Then he lost 23 on tuesday.
So he had 58 - 23 = 35 golf balls.
Then he lost 2 more on wednesday.
So he must have 35 - 2 = 33 golf balls.
The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step
First Olivia has 23 dollars.
Then she bought five bagels for 3 dollars each.
We know 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars.
So she has 23 - 15 = 8 dollars left.
The answer is 8.

"""

recall_infer_prompt ="""Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step
First there are 15 trees.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6 trees.
The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step
First there are 3 cars.
Then 2 more cars arrive.
So there must be 3 + 2 = 5 cars.
The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let's think step by step
First Leah had 32 chocolates and her sister had 42 chocolates.
So in total they had 32 + 42 = 74 chocolates.
Then they ate 35 chocolates.
So there must be 74 - 35 = 39 chocolates.
The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step
First Jason had 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8 lollipops.
The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step
First Shawn has 5 toys.
Then he got 2 toys each from his mom and dad.
So he must have 5 + 4 = 9 toys.
The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's think step by step
First there were 9 computers.
Then for each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
So there must be in total 9 + 20 = 29 computers.
The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step
First Michael started with 58 golf balls.
Then he lost 23 on tuesday.
So he had 58 - 23 = 35 golf balls.
Then he lost 2 more on wednesday.
So he must have 35 - 2 = 33 golf balls.
The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step
First Olivia has 23 dollars.
Then she bought five bagels for 3 dollars each.
Then 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars.
So she has 23 - 15 = 8 dollars left.
The answer is 8.

"""

my_cot_prompts_without_equations = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step
First there are 15 trees.
Then there were 21 trees after some more were planted.
So there must have been 6 trees.
The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step
First there are 3 cars.
Then 2 more cars arrive.
So there must be 5 cars.
The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let's think step by step
First Leah had 32 chocolates and her sister had 42 chocolates.
So in total they had 74 chocolates.
Then they ate 35 chocolates.
So there must be 39 chocolates.
The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step
First Jason had 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 8 lollipops.
The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step
First Shawn has 5 toys.
Then he got 2 toys each from his mom and dad.
So he must have 9 toys.
The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's think step by step
First there were 9 computers.
Then for each of 4 days, 5 more computers were added.
So 20 computers were added.
So there must be in total 29 computers.
The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step
First Michael started with 58 golf balls.
Then he lost 23 on tuesday.
So he had 35 golf balls.
Then he lost 2 more on wednesday.
So he must have 33 golf balls.
The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step
First Olivia has 23 dollars.
Then she bought five bagels for 3 dollars each.
We know 5 bagels for 3 dollars each will be 15 dollars.
So she has 8 dollars left.
The answer is 8.

"""

my_cot_prompts_without_text = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
21 - 15 = 6
The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
3 + 2 = 5
The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
32 + 42 = 74
74 - 35 = 39
The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
20 - 12 = 8
The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
5 + 4 = 9
The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
5 * 4 = 20 
9 + 20 = 29
The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
58 - 23 = 35
35 - 2 = 33.
The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
5 * 3 = 15
23 - 15 = 8
The answer is 8.

"""

my_cot_prompts_incorrect = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees.
How many trees did the grove workers plant today?
Let’s think step by step
First there are 15 trees.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 1 trees.
The answer is 1.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let’s think step by step
First there are 3 cars.
Then 2 more cars arrive.
So there must be 3 + 2 = 3 cars.
The answer is 3.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let’s think step by step
First Leah had 32 chocolates and her sister had 42 chocolates.
So in total they had 32 + 42 = 12 chocolates.
Then they ate 35 chocolates.
So there must be 12 - 35 = 45 chocolates.
The answer is 45.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to
Denny?
Let’s think step by step
First Jason had 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 2 lollipops.
The answer is 2.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let’s think step by step
First Shawn has 5 toys.
Then he got 2 toys each from his mom and dad.
So he must have 5 + 4 = 6 toys.
The answer is 6.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How
many computers are now in the server room?
Let’s think step by step
First there were 9 computers.
Then for each of 4 days, 5 more computers were added.
So 5 * 4 = 22 computers were added.
So there must be in total 9 + 22 = 23 computers.
The answer is 22.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Let’s think step by step
First Michael started with 58 golf balls.
Then he lost 23 on Tuesday.
So he had 58 - 23 = 14 golf balls.
Then he lost 2 more on Wednesday.
So he must have 14 - 2 = 23 golf balls.
The answer is 23.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let’s think step by step
First Olivia has 23 dollars.
Then she bought five bagels for 3 dollars each.
We know 5 bagels for 3 dollars each will be 5 * 3 = 23 dollars.
So she has 23 - 23 = 9 dollars left.
The answer is 9.

"""

my_cot_prompts_incorrect_ood = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let’s think step by step
First there are 15 trees.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = Dawson trees.
The answer is Dawson.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let’s think step by step
First there are 3 cars.
Then 2 more cars arrive.
So there must be 3 + 2 = Russel cars.
The answer is Russel.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let’s think step by step
First Leah had 32 chocolates and her sister had 42 chocolates.
So in total they had 32 + 42 = Cedric chocolates.
Then they ate 35 chocolates.
So there must be Cedric - 35 = Hendricks chocolates.
The answer is Hendricks.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let’s think step by step
First Jason had 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = Edison lollipops.
The answer is Edison.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let’s think step by step
First Shawn has 5 toys.
Then he got 2 toys each from his mom and dad.
So he must have 5 + 4 = Kaylin toys.
The answer is Kaylin.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let’s think step by step
First there were 9 computers.
Then for each of 4 days, 5 more computers were added.
So 5 * 4 = Mcgowan computers were added.
So there must be in total 9 + Mcgowan = Damarion computers.
The answer is Damarion.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Let’s think step by step
First Michael started with 58 golf balls.
Then he lost 23 on Tuesday.
So he had 58 - 23 = Jemima golf balls.
Then he lost 2 more on Wednesday.
So he must have Jemima - 2 = Astrid golf balls.
The answer is Astrid.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let’s think step by step
First Olivia has 23 dollars.
Then she bought five bagels for 3 dollars each.
We know 5 bagels for 3 dollars each will be 5 * 3 = Gallagher dollars.
So she has 23 - Gallagher = Baily dollars left.
The answer is Baily.

"""

my_cot_prompts_add_only = """Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let’s think step by step
First there are 3 cars.
Then 2 more cars arrive.
So there must be 3 + 2 = 5 cars.
The answer is 5.

Question: Paddington has 40 more goats than Washington. If Washington has 140 goats, how many goats do they have in total?
Let’s think step by step
First Paddington has 40 more goats than Washington.
We know Washington has 140 goats.
So Paddington has 40 + 140 = 180 goats.
So they have 180 + 140 = 320 goats in total.
The answer is 320.

Question: Christina has 3 snakes. 1 snake is 2 feet long. Another snake is 16 inches long. The last snake is 10 inches long. How many inches are all of her snakes combined?
Let’s think step by step
First Christina has 3 snakes.
Then 1 snake is 2 feet long.
We know 1 foot is 12 inches.
So 2 feet is 12 + 12 = 24 inches.
Then another snake is 16 inches long.
Then the last snake is 10 inches long.
So all of her snakes combined are 24 + 16 + 10 = 50 inches.
The answer is 50.

Question: Bush and Matt are brothers. Bush is younger than Matt by 3 years. This year Bush will be 12 years old. What will be Matt’s age 10
years from now?
Let’s think step by step
First Bush is younger than Matt by 3 years.
We know Bush will be 12 years old this year.
So Matt will be 12 + 3 = 15 years old this year.
Then Matt’s age 10 years from now will be 15 + 10 = 25 years old.
The answer is 25.

Question: Jeremy listened to five more songs yesterday than today. Yesterday, he listened to nine songs. How many songs did Jeremy listen to in two days?
Let’s think step by step
First Jeremy listened to 9 songs yesterday.
Then he listened to 5 more songs yesterday than today.
So he listened to 9 + 5 = 14 songs today.
So he listened to 9 + 14 = 23 songs in two days.
The answer is 23.

Question: Jar A has 28 marbles. Jar B has 12 more marbles than jar A. Jar C has as many marbles as jar B. How many marbles are there
altogether?
Let’s think step by step
First Jar A has 28 marbles.
Then Jar B has 12 more marbles than jar A.
So Jar B has 28 + 12 = 40 marbles.
Then Jar C has as many marbles as jar B.
So Jar C has 40 marbles.
So there are 28 + 40 + 40 = 108 marbles altogether.
The answer is 108.

Question: Marion received 20 more turtles than Mia at the animal rescue center. If Mia received 40 turtles, how many turtles did they receive together?
Let’s think step by step
First Marion recieved 20 more turtles than Mia.
We know Mia received 40 turtles.
So Marion received 20 + 40 = 60 turtles.
So together they received 60 + 40 = 100 turtles.
The answer is 100.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let’s think step by step
First Shawn has 5 toys.
Then he got 2 toys each from his mom and dad.
So he must have 5 + 4 = 9 toys.
The answer is 9.

"""

my_cot_prompts_mult_only = """Question: Super Clean Car Wash Company cleans 80 cars per day. They make $5 per car washed. How much money will they make in 5 days?
Let’s think step by step
First Super Clean Car Wash Company cleans 80 cars per day.
Then they make 5 dollars per car washed.
So they make 80 x 5 = 400 dollars per day.
So they make 400 x 5 = 2000 dollars in 5 days.
The answer is 2000.

Question: A farmer has 5 cows. Each cow has 4 legs. How many legs do the cows have in total?
Let’s think step by step
First a farmer has 5 cows.
Then each cow has 4 legs.
So the cows have 5 x 4 = 20 legs in total.
The answer is 20.

Question: Sam watches two movies each day. Each movie is 2 hours long. How many minutes does Sam spend watching movies in 5 days?
Let’s think step by step
First Sam watches two movies each day.
Then each movie is 2 hours long.
We know 1 hour is 60 minutes.
So 2 hours is 60 x 2 = 120 minutes.
So Sam spends 120 x 2 = 240 minutes watching movies each day.
So Sam spends 240 x 5 = 1200 minutes watching movies in 5 days.
The answer is 1200.

Question: Carla has 3 bags. Each bag has 5 apples. How many apples does Carla have in total?
Let’s think step by step
First Carla has 3 bags.
Then each bag has 5 apples.
So Carla has 3 x 5 = 15 apples in total.
The answer is 15.

Question: James takes 20 units per semester at community college. If each unit costs $50 how much does he pay for 2 semesters?
Let’s think step by step
First James takes 20 units per semester at community college.
Then each unit costs 50 dollars.
So he pays 20 x 50 = 1000 dollars per semester.
So he pays 1000 x 2 = 2000 dollars for 2 semesters.
The answer is 2000.

Question: In a jar that has 50 ants, the number of ants in the jar doubles each hour. How many ants will be in the jar after 5 hours?
Let’s think step by step
First there are 50 ants in the jar.
Then the number of ants in the jar doubles each hour.
So there will be 50 x 2 = 100 ants in the jar after 1 hour.
So there will be 100 x 2 = 200 ants in the jar after 2 hours.
So there will be 200 x 2 = 400 ants in the jar after 3 hours.
So there will be 400 x 2 = 800 ants in the jar after 4 hours.
So there will be 800 x 2 = 1600 ants in the jar after 5 hours.
The answer is 1600.

Question: Mark loves to see shows in theaters. He decided to visit the theater at least once a week. One performance lasts 3 hours. The price of the ticket depends on the time spent in the theater and stands at $5 for each hour. How much will Mark spend on visits to the theater in 6 weeks?
Let’s think step by step
First Mark decided to visit the theater at least once a week.
Then one performance lasts 3 hours.
We know the price of the ticket depends on the time spent in the theater and stands at 5 dollars for each hour.
So the price of the ticket for one performance is 5 x 3 = 15 dollars.
So Mark will spend 15 x 6 = 90 dollars on visits to the theater in 6 weeks.

The answer is 90.
Question: A sixty bulb watt uses 60 watts of power each day. If Allyn has 40 such bulbs in his house and pays an electricity bill of twenty cents per power watt used, calculate Allyn’s total monthly expenses on electricity in June.
Let’s think step by step
First a sixty bulb watt uses 60 watts of power each day.
Then Allyn has 40 such bulbs in his house.
So Allyn has 40 x 60 = 2400 watts of power each day.
Then Allyn pays an electricity bill of twenty cents per power watt used.
So Allyn pays 2400 x 0.2 = 480 dollars per day.
So Allyn pays 480 x 30 = 14400 dollars per month.
The answer is 14400.

"""