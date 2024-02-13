---
hidden: true
---


All four lines
```python
{% include _includes/includesnippet filename='_includes/testfile.py' %}
```

two lines
```python
{% include _includes/includesnippet filename='_includes/testfile.py' count=2 %}
```

start at line 2
```python
{% include _includes/includesnippet filename='_includes/testfile.py' start=2 %}
```

start at line 1, but do two lines
```python
{% include _includes/includesnippet filename='_includes/testfile.py' start=1 count=2 %}
```

get main
```python
{% include _includes/includesnippet filename='_includes/testfile.py' starttext='main()' endtext='return' %}
```

get comment to return
```python
{% include _includes/includesnippet filename='_includes/testfile.py' starttext='A comment' endtext='return' %}
```

get comment to assignment
```python
{% include _includes/includesnippet filename='_includes/testfile.py' starttext='A comment' endtext='p = 2' %}
```

