/*
* @Author: zHanami
* @Date:   2017-12-16 11:50:01
* @Last Modified by:   Marte
* @Last Modified time: 2017-12-16 21:49:10
*/

% 7.Solution of Nonlinear Equations

% bisect
function x = bisect(f,a,b,tol)
if nargin<4,
    tol=1e-12;
end
fa=feval(f,a); fb=feval(f,b);
while abs(a-b)>tol,
    x=(a+b)/2;
    fx=feval(f,x);
    if sign(fx)==sign(fa),
        a=x; fa=fx;
    elseif sign(fx)==sign(fb),
        b=x; fb=fx;
    else return;
    end
end

% newtown downhill
function [x,it,convg] = newton(x0,f,g,maxit,tol)
% find the zero of function f, with gradient g provided
% Usage: [x,it,convg] = newton(x0,f,g,maxit,tol)
% f: original function, g: derivation of original function
% maxit: the maximum iterations, tol: precision
if nargin<5,
    tol = 1e-10;
    if nargin<4,
        maxit = 100;
    end
end
x = x0;
fx = feval(f,x);
convg = 0;
it = 1;
while ~convg,
    it = it + 1;
    if norm(fx)<=tol,
        fprintf('Newton Iteration successes!!\n');
        convg = 1;
        return;
    end
    d = - feval(g,x) \ fx;
    lambda = 1;
    lsdone = 0;
    while ~lsdone,
        xn = x + lambda * d;
        fn = feval(f,xn);
        if abs(fn)<abs(fx),
            lsdone = 1;
        else
            lambda = 1/2 * lambda;
            if lambda<=eps,
                convg = -1;
                error('line search fails!!');
            end
        end
    end
    x = xn;
    fx = fn;
    if it > maxit,
        convg = 0;
        error('Newton method needs more iterations.!!');
    end
end


% 8.The calculation of eigenvalues and eigenvectors of a matrix

% power method
function [t,y] = eigIPower(a,xinit,ep)
% a: the matrix A xinit: v0
v0 = xinit;
[tv,ti] = max(abs(v0));
lam0 = v0(ti);
u0 = v0/lam0;
flag = 0;
while (flag==0)
    v1 = a*u0;
    [tv,ti] = max(abs(v1));
    lam1 = v1(ti);
    u0 = v1/lam1;
    err = abs(lam0-lam1);
    if (err<=ep)
        flag = 1;
    end
    lam0 = lam1;
end
t = lam1;
y = u0;

% The inverse power of the translation of the origin
function [t,y] = invPower(a,xinit,ep,p)
v0 = xinit;
[tv,ti] = max(abs(v0));
lam0 = v0(ti);
u0 = v0/lam0;
n = length(a);
flag = 0;
while (flag==0)
    v1=inv(a-p*eye(n))*u0;
    [tv,ti] = max(abs(v1));
    lam1 = v1(ti);
    u0 = v1/lam1;
    err = abs(1./lam1-1./lam0);
    if (err<=ep)
        flag = 1;
    end
    lam0 = lam1;
end
t = 1./lam1+p;
y = u0;


% 9.The numerical solution of the initial boundary value problem for ordinary differential equations

% Euler formula
function [x,y] = odeeuler(f,y0,a,b,n)
% f: derivation of original function
y(1) = y0;
h = (b-a)/n;
x = a:h:b;
for i = 1:n,
    y(i+1) = y(i) + h*feval(f,x(i),y(i));
end

% optimization of Euler formula
function [x,y] = odeIEuler(f,y0,a,b,n)
y(1) = y0;
h = (b-a)/n;
x = a:h:b;
for i=1:n,
    yp = y(i) + h*feval(f,x(i),y(i));
    yc = y(i) + h*feval(f,x(i+1),yp);
    y(i+1) = 0.5 * (yp+yc);
end

% rk2
function [x,y] = rk2(dfun,a,b,y0,h)
x = a:h:b;
n = length(x);
y(1) = y0;
for k = 2:n,
    k1 = h * feval(dfun,x(k-1),y(k-1));
    k2 = h * feval(dfun,x(k-1)+h/2,y(k-1)+k1/2);
    y(k) = y(k-1) + 1/2 * (k1+k2);
end

% rk4
function [x,y] = rk4(dfun,a,b,y0,h)
x = a:h:b;
n = length(x);
y(1) = y0;
for k = 2:n,
    k1 = h * feval(dfun,x(k-1)),y(k-1));
    k2 = h * feval(dfun,x(k-1)+h/2,y(k-1)+k1/2);
    k3 = h * feval(dfun,x(k-1)+h/2,y(k-1)+k2/2);
    k4 = h * feval(dfun,x(k-1)+h,y(k-1)+k3);
    y(k)=y(k-1)+(k1+2*k2+2*k3+k4)/6;
end