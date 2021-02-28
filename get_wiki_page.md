```
! pip install wikipedia
import wikipedia
page = wikipedia.page(wikipedia.search('BarackObama')[0])
dir(page)
page.summary
page.content
```