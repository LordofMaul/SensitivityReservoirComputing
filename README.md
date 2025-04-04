<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>



<h1>Sensitivity on Reservoir Computing wrt. training data</h1>
<h2>Generating Data by solivng Hénon-Heiles-System</h2>
Hamiltonian given by:
$$H=\frac{1}{2}(x^2+y^2+\dot x^2+\dot y^2)+x^2y-\frac{y^3}{3}$$
Solving using Runge-Kutta-4 method
