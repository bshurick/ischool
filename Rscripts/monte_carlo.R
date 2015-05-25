# R exercise:
#   
#   We are going to use Monte Carlo simulation to calculate the area of two shapes repeatedly and then use a t test to see whether these shapes have similar or different areas. 
# 
# The two shapes are 
# (a) circle centered on origin (0,0) and with a radius of 1.
# (b) a rhombus (diamond) with corners on-in counterclockwise order: (1.6,0), (0,-1), (-1.6,0),(0,1)
# 
# Here is the basic idea:
#   -generate a bunch (say 100 times) of x and y coordinates within a shape that is bigger than the shape you are interested in and whose area you already know. The best choice is a square and the way to generate the coordinates is though a uniform distribution.
# 
# -for each shape, see whether each randomly generated point is inside or outside your shape of interest. add them up. The area is: (fraction of points within)*(area of the mother square)
# 
# -repeat a bunch of times (say, 20 times) for each shape.
# 
# ========
#   1-what is an appropriate test to use here? 
# -make a theoretical case. first answer this: what is the distribution of each of the area vectors we made?
# 2-what is the result? 
# 3-Do the whole thing a number of times but for circles (so that the null is  true). How many times the procedure rejects the null?
# 
# 
# Hint 1: shapes are symmetric and it is easier to work with one quarter of their size (in one quadrant) and then multiply by 4. This changes the meaning of within to underneath: is the random point underneath the curve or not.
# 
# Hint 2: for the circle, the formula is y=+/- sqrt(1-x^2)

### area under y=fun(x) is calculated in the first quadrant
area = function(fun,n=10, maxx=1, maxy=1){
  x=runif(n,0,maxx)
  y=runif(n,0,maxy)
  yy = fun(x)
  a = sum(y<yy)/n *4 * maxx* maxy
  return(a)
}

### doing it 20 times for the circle and the rhombus
### the numbers are small so the doesn't have much power
a1 = replicate(20, area(function(x) sqrt(1-x^2) ) )
a2 = replicate(20, area(function(x) 1-x/1.6,maxx=1.6) )
t.test(a1,a2)

### doing a t-test for two similar circles 
### The null hyp. of no-difference is correct by design
res = sapply(1:1000, function(i){
  a1 = replicate(20, area(function(x) sqrt(1-x^2) ) )
  a2 = replicate(20, area(function(x) sqrt(1-x^2) ) )
  t.test(a1,a2)$p.value <0.05
})
print( mean(res) )

