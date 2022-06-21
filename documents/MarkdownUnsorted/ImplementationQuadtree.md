
# Efficient implementation of a Quadtree (for 2D collision detection)

[Stackoverflow: Efficient implementation of Quadtrees (for 2D collision detection)](https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det)

## Other possibility (than using Quadtrees)

Using a **Grid hierarchy**, like

* a coarse grid for the world
* a finer grid for a region
* an even finer grid for a sub-region

meaning 3 fixed levels of dense grids (no trees involved), with row-based optimizations so that a row withoug entities will be deallocated and turn into a null pointer.


## Fundamentals of Quadtrees

**Quadtrees are non-fixed resolutin grids, they adapt the resolution based on some criteria, while subdiving/splitting into 4 child cells to increase resolution.**

## Implemenation in C

See [GitHub: quadtree](https://github.com/Antymon/quadtree)


## Possible improvements regarding Implementations of Quadtrees

### Node representation

```c
// Represents a node in the quadtree.
struct QuadNode
{
    // Points to the first child if this node is a branch or the first
    // element if this node is a leaf.
    int32_t first_child;

    // Stores the number of elements in the leaf or -1 if it this node is
    // not a leaf.
    int32_t count;
};
```

**Possible improvement**: Memory reduction

* **reduce the memory size** to a reasonable small amount
* **no bounding boxes/rectangles** (AABBs) stored within the node representation, compute them on the fly
	* just store them **once in the root**
	* seems to be more expensive to be computed on the fly, but reducing memory usage of the nodes can proportianlly reduce cache misses when traversing the tree, which tend to be more significant
	* not reasonable for data structures that don't split evenly like Kd-trees and BVHs

### Traversel

Traversel looks like

```c
static QuadNodeList find_leaves(const Quadtree& tree, const QuadNodeData& root, const int rect[4])
{
    QuadNodeList leaves, to_process;
    to_process.push_back(root);
    while (to_process.size() > 0)
    {
        const QuadNodeData nd = to_process.pop_back();

        // If this node is a leaf, insert it to the list.
        if (tree.nodes[nd.index].count != -1)
            leaves.push_back(nd);
        else
        {
            // Otherwise push the children that intersect the rectangle.
            const int mx = nd.crect[0], my = nd.crect[1];
            const int hx = nd.crect[2] >> 1, hy = nd.crect[3] >> 1;
            const int fc = tree.nodes[nd.index].first_child;
            const int l = mx-hx, t = my-hx, r = mx+hx, b = my+hy;

            if (rect[1] <= my)
            {
                if (rect[0] <= mx)
                    to_process.push_back(child_data(l,t, hx, hy, fc+0, nd.depth+1));
                if (rect[2] > mx)
                    to_process.push_back(child_data(r,t, hx, hy, fc+1, nd.depth+1));
            }
            if (rect[3] > my)
            {
                if (rect[0] <= mx)
                    to_process.push_back(child_data(l,b, hx, hy, fc+2, nd.depth+1));
                if (rect[2] > mx)
                    to_process.push_back(child_data(r,b, hx, hy, fc+3, nd.depth+1));
            }
        }
    }
    return leaves;
}
```

**Possible improvement**: Floating-Point

* Do not use floating-point (for spatial indexes, ...)
* Even use integers for floating-point inputs 

### Contiguous children

```c
struct QuadNode
{
    int32_t first_child;
    ...
};
```

**Possible improvement**: Contiguosity 

* No need to store an array of children, because all 4 children are contiguous
	* reduces cache misses on traversal
	* shrinks nodes (further reduces cache misses)
	* **consequently**: splitting a parent means allocating for 4 children, even if some of the children are empty
		* trade-off is worth it (in respect to performance)
		* deallocating children means deallocating **all 4 children at a time** using an indexed free list 
			* consequently usually no heap allocations or deallocations during the simulation	 

```
first_child+0 = index to 1st child (TL)
first_child+1 = index to 2nd child (TR)
first_child+2 = index to 3nd child (BL)
first_child+3 = index to 4th child (BR)
```

![Deallocation of children](Images/DeallocateChildren.png)

### Deferred cleaning

**Possible improvement:** deferred cleanup

* Don't update the quadtree's structure right away, just traverse to the child node(s) to be removed and remove the element and don't bother to do more even if the leaves become empty
	* unnecesarily removing children only to add them right back when another element moves into that quadrant is avoided 
* The cleanup method never removes the root

Implementation looks like:

```c
void Quadtree::cleanup()
{
    // Only process the root if it's not a leaf.
    SmallList<int> to_process;
    if (nodes[0].count == -1)
        to_process.push_back(0);

    while (to_process.size() > 0)
    {
        const int node_index = to_process.pop_back();
        QuadNode& node = nodes[node_index];

        // Loop through the children.
        int num_empty_leaves = 0;
        for (int j=0; j < 4; ++j)
        {
            const int child_index = node.first_child + j;
            const QuadNode& child = nodes[child_index];

            // Increment empty leaf count if the child is an empty 
            // leaf. Otherwise if the child is a branch, add it to
            // the stack to be processed in the next iteration.
            if (child.count == 0)
                ++num_empty_leaves;
            else if (child.count == -1)
                to_process.push_back(child_index);
        }

        // If all the children were empty leaves, remove them and 
        // make this node the new empty leaf.
        if (num_empty_leaves == 4)
        {
            // Push all 4 children to the free list.
            nodes[node.first_child].first_child = free_node;
            free_node = node.first_child;

            // Make this node the new empty leaf.
            node.first_child = -1;
            node.count = 0;
        }
    }
}
```

which is called at the end of every frame, when removing all agents/particles.

### Moving elements

Straightforward:

* remove element
* move element
* reinsert to quadtree

### Singly-Linked Index Lists for Elements

**Possible improvement:** Singly-linked index lists for elements

* transfer elements from split parents to new leaves by just changing a few integers

Use the representation:

```c
// Represents an element in the quadtree.
struct QuadElt
{
    // Stores the ID for the element (can be used to
    // refer to external data).
    int id;

    // Stores the rectangle for the element.
    int x1, y1, x2, y2;
};

// Represents an element node in the quadtree.
struct QuadEltNode
{
    // Points to the next element in the leaf node. A value of -1 
    // indicates the end of the list.
    int next;

    // Stores the element index.
    int element;
};
```

**Possible improvement:** Avoid heap allocation (use the stack, whenever possible)

* e.g. for C++: use `SmallList<T>' instead of 'vector<T>` for temporary stack of nodes, since no heap allocation is involved (until more than 128 elements are inserted)


### Tree representation

**Possible improvement:** Elements should be stored in the tree and leaf nodes should index or point to those elements!

Representation of the quadtree itself:

```c
struct Quadtree
{
    // Stores all the elements in the quadtree.
    FreeList<QuadElt> elts;

    // Stores all the element nodes in the quadtree.
    FreeList<QuadEltNode> elt_nodes;

    // Stores all the nodes in the quadtree. The first node in this
    // sequence is always the root.
    std::vector<QuadNode> nodes;

    // Stores the quadtree extents.
    QuadCRect root_rect;

    // Stores the first free node in the quadtree to be reclaimed as 4
    // contiguous nodes at once. A value of -1 indicates that the free
    // list is empty, at which point we simply insert 4 nodes to the
    // back of the nodes array.
    int free_node;

    // Stores the maximum depth allowed for the quadtree.
    int max_depth;
};
```

All nodes are stored in contiguously in an array (`std::vector<QuadNode>`) along with the elements and element nodes (in `FreeList<T>`).

#### FreeList<T>

A `FreeList` data structure which is basically an array (and random-access sequence) that lets you remove elements from anywhere in constant-time (leaving holes behind which get reclaimed upon subsequent insertions in constant-time). Here's a simplified version which doesn't bother with handling non-trivial data types (doesn't use placement new or manual destruction calls):

```
/// Provides an indexed free list with constant-time removals from anywhere
/// in the list without invalidating indices. T must be trivially constructible 
/// and destructible.
template <class T>
class FreeList
{
public:
    /// Creates a new free list.
    FreeList();

    /// Inserts an element to the free list and returns an index to it.
    int insert(const T& element);

    // Removes the nth element from the free list.
    void erase(int n);

    // Removes all elements from the free list.
    void clear();

    // Returns the range of valid indices.
    int range() const;

    // Returns the nth element.
    T& operator[](int n);

    // Returns the nth element.
    const T& operator[](int n) const;

private:
    union FreeElement
    {
        T element;
        int next;
    };
    std::vector<FreeElement> data;
    int first_free;
};

template <class T>
FreeList<T>::FreeList(): first_free(-1)
{
}

template <class T>
int FreeList<T>::insert(const T& element)
{
    if (first_free != -1)
    {
        const int index = first_free;
        first_free = data[first_free].next;
        data[index].element = element;
        return index;
    }
    else
    {
        FreeElement fe;
        fe.element = element;
        data.push_back(fe);
        return static_cast<int>(data.size() - 1);
    }
}

template <class T>
void FreeList<T>::erase(int n)
{
    data[n].next = first_free;
    first_free = n;
}

template <class T>
void FreeList<T>::clear()
{
    data.clear();
    first_free = -1;
}

template <class T>
int FreeList<T>::range() const
{
    return static_cast<int>(data.size());
}

template <class T>
T& FreeList<T>::operator[](int n)
{
    return data[n].element;
}

template <class T>
const T& FreeList<T>::operator[](int n) const
{
    return data[n].element;
}

```


#### Maximum tree depth

**Possible improvement:** Prevent the tree from subdividing too much

* specify a maximum depth of the tree

#### Queries

**Possible improvement:** Cache friendly queries


Use:

```c
traversed = {}
gather quadtree leaves
for each leaf in leaves:
{
     for each element in leaf:
     {
          if not traversed[element]:
          {
              use quad tree to check for collision against other elements
              traversed[element] = true                  
          }
     }
}
```

Instead of:

```c
for each element in scene:
     use quad tree to check for collision against other elements
```

* it helps make sure that we descend the same paths down the quadtree throughout the loop. That helps keep things much more cache-friendly. Also if after attempting to move the element in the time step, it's still encompassed entirely in that leaf node, we don't even need to work our way back up again from the root (we can just check that one leaf only)

## Things to avoid


### Full-blown containers

Doing something like

```c
struct Node
{
     ...

     // Stores the elements in the node.
     List<Element> elements; //e.g. std::vector
};
```

which allocates and frees it owns memory.

These containers are very efficiently implemented to store a large number of elements, but are extremely inefficient for instantiating a bootload of them to only store a few elements in each one of them, since the containers metadata tends to be quite explosive.


## Loose quadtree

See [Stackoverflow](https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det)








  