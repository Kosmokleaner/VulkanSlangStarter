#pragma once
#include <vector>
#include <new>		// placement new, used by CObjManager to run T's constructor/destructor explicitly
#include <utility>	// std::forward, used by emplace_back

// generalization handle 

// weak reference into CObjManager
class CObjHandle
{
public:
	CObjHandle()
		: Seed(-1)
		, Index(0)
	{
	}

	bool IsValid() const
	{
		return Seed != (uint32)-1;
	}

	// assignment operator
	CObjHandle& operator=(const CObjHandle& rhs)
	{
		if (this != &rhs)
		{
			Seed = rhs.Seed;
			Index = rhs.Index;
		}
		return *this;
	}

	// dangerous e.g. uint32 index = hnd;	
//	operator bool() const { return IsValid(); }

	// comparison operators
	bool operator==(const CObjHandle& rhs) const { return Seed == rhs.Seed && Index == rhs.Index; }
	bool operator!=(const CObjHandle& rhs) const { return !(*this == rhs); }

	// for std::set<> / std::map<>
	bool operator<(const CObjHandle& rhs) const { return getRaw() < rhs.getRaw(); }

	void setRaw(const uint64 inRaw)
	{
		Seed = (uint32)(inRaw >> 32);
		Index = (uint32)(inRaw & 0xffffffff);
	}

	uint64 getRaw() const
	{
		return ((uint64)Seed << 32) | (uint64)Index;
	}
	
//private:

	// to validate if the Index is still valid, rename to version?
	uint32 Seed;
	// Index in CObjManager::Elements, could be made 16bit
	uint32 Index;
};

// handle with type
class TypedHandle : public CObjHandle
{
public:
	TypedHandle() : CObjHandle() {}
	TypedHandle(CObjHandle hnd, uint32 inType) : CObjHandle(hnd), type(inType) {}

	// 0:entity, 1:material, 2:texture, 3:mesh, 4:light, 5:camera, ..., -1:not set, todo: register,
	// store in CObjHandle, consider 16byte GUID and bulk operation
	uint32 type = -1;
};

static_assert(sizeof(TypedHandle) == sizeof(CObjHandle) + sizeof(uint32));


class CTexHandle : public CObjHandle {};

// object type
class CMakeHandle : public CObjHandle {};

// fixed size, no reallocations. T can be a pointer or an object; T's constructor/destructor run on
// add/remove (see emplace_back(), RemoveByIndex())
// @param THandle e.g. CObjHandle, CTexHandle
// @param TSeedEnd only for testing
template <class THandle, class T, int TSeedEnd = -1>
class CObjManager
{
public:

	// O(1)
	CObjManager()
		: UsedCount(0)
		, FreeCount(0)
	{
	}

	// should only be called once
	// O(n)
	// @param Size number of slots; how often a single slot can be reused is limited by the "Seed" type (uint32) or TSeedEnd
	void InitOnce(uint32 Size)
	{
		ASSERT(Size > 0);
		ASSERT(!Elements.size());
		ASSERT(!UsedCount);

		Elements.resize(Size);

		for (uint32 i = 0; i < Size; ++i)
		{
			Elements[i].IndexInUsed = -1;		// not needed but makes debugging easier
			Elements[i].Seed = 0;
		}

		UsedAndFree.resize(Size);

		for (uint32 i = 0; i < Size; ++i)
		{
			UsedAndFree[i] = i;
		}

		UsedCount = 0;
		FreeCount = Size;
	}

	// O(n), Seeds are not reset allowing existing CObjHandles to becomes invalid without iterating them
	void Clear()
	{
		// could be optimized
		while (UsedCount)
		{
			RemoveByIndex(UsedAndFree[0]);
		}

	}

	// O(1)
	// @return could be invalid if pool is full, test with GetElementPtr()
	THandle push_back(const T& data)
	{
		return emplace_back(data);
	}

	// Constructs T in its final slot directly from `args`, instead of push_back()'s "build a
	// temporary, then copy it in" -- required for a T that owns a registration-based resource (e.g.
	// a Buffer member, auto-registering itself with g_EventManager via IAppInit, see EventManager.h)
	// where the compiler-generated copy constructor a temporary would go through silently does NOT
	// re-register the copy. That copy still looks fully valid right up until shutdown's
	// callAppDeInit() sweep -- driven by that same registration list -- skips it, leaving its GPU
	// buffer(s) never deinit()'d.
	template <class... Args>
	THandle emplace_back(Args&&... args)
	{
		THandle Ret;

		if (!FreeCount)
		{
			// call InitOnce()
			ASSERT(UsedCount);

			// no free elements, we fail
			return Ret;
		}

		uint32 ElementIndex = UsedAndFree[UsedCount];

		struct TAndGuid& el = Elements[ElementIndex];

		// -1 means seed depleted, it should no longer be in the free list
		ASSERT(el.Seed != (uint32)TSeedEnd);

		// el.t is always a valid, constructed placeholder here (either the default object
		// from InitOnce()'s vector::resize(), or the one placement-constructed by
		// RemoveByIndex() below) -- destroy it and construct the real object fresh instead
		// of assigning over it, so T's constructor observably runs once per reuse
		el.t.~T();
		new (&el.t) T(std::forward<Args>(args)...);

		el.IndexInUsed = UsedCount;

		Ret.Index = ElementIndex;
		Ret.Seed = el.Seed;

		++UsedCount;
		--FreeCount;

		VerifyInternals();

		return Ret;
	}

	// does nothing if element doesn't exist
	void Remove(const CObjHandle& Hnd)
	{
		ASSERT(Hnd.Index < UsedAndFree.size());

		if (!Hnd.IsValid())
		{
			return;
		}

		TAndGuid& el = Elements[Hnd.Index];

		if (el.Seed != Hnd.Seed)
		{
			VerifyInternals();
			return;
		}

		RemoveByIndex(Hnd.Index);

		VerifyInternals();
	}

	// @param Index 0 .. GetElementCount()-1 in Elements[]
	void RemoveByIndex(uint32 Index)
	{
		ASSERT(Index < UsedAndFree.size());

		TAndGuid& el = Elements[Index];

		ASSERT(el.IndexInUsed != -1);

		el.Seed++;

		--UsedCount;

		// end the removed object's lifetime now, then immediately construct a fresh
		// placeholder in its place so el.t stays a valid, once-constructed T at all times --
		// needed so a later CObjManager destruction (vector's own ~TAndGuid() sweep) never
		// runs T's destructor on already-destructed memory
		el.t.~T();
		new (&el.t) T();

		// can it be reused or is the index depleted
		if (el.Seed != (uint32)TSeedEnd)
		{
			// it can be reused
			++FreeCount;

			SwapInUsedAndUsed(el.IndexInUsed, UsedCount);

			el.IndexInUsed = -1;		// not needed but makes debugging easier

			VerifyInternals();
		}
		else
		{
			// put in dead list
			SwapInUsedAndUsed(el.IndexInUsed, UsedCount + FreeCount);
			VerifyInternals();
		}
	}

	// O(1)
	// @return 0 if the element no longer exists
	T* GetElementPtr(const THandle& Hnd)
	{
		if (Hnd.Index < Elements.size())
		{
			TAndGuid& el = Elements[Hnd.Index];

			if (Hnd.Seed == el.Seed)
				return &el.t;
		}

		return 0;
	}

	// @return iterator
	uint32 GetIteratorFromHandle(const THandle hnd)
	{
		TAndGuid& el = Elements[hnd.Index];

		return el.IndexInUsed;
	}
	
	// O(1)
	// useful to iterate all elements, see GetElementByIterator()
	uint32 GetElementCount() const { return UsedCount; }
	// todo: remove this or the other
	uint32 size() const { return UsedCount; }
	bool empty() const { return UsedCount == 0; }

	// Total slots (used + free + dead), i.e. Elements.size() -- fixed at whatever InitOnce() was
	// given. THandle::Index (the position in Elements[], what GetElementPtr()/GetHandleByIndex()
	// index with) is always < this, even though it's only < GetElementCount() when nothing has
	// ever been removed-and-reused (see GetHandleByIterator()'s own comment on why those two
	// diverge) -- a CPU-side array meant to be indexed by THandle::Index (e.g. matching a GPU id
	// built the same way) must be sized by this, not GetElementCount(), or an index recycled from
	// Clear()-ing a larger previous population writes past the end.
	uint32 GetCapacity() const { return (uint32)Elements.size(); }

	// O(1)
	// @param Iterator .. 0..GetElementCount()-1, assert if outside of the range
	// @return always valid
	T& GetElementByIterator(uint32 Iterator)
	{
		ASSERT(Iterator < UsedCount);

		uint32 ElementIndex = UsedAndFree[Iterator];

		TAndGuid& el = Elements[ElementIndex];

		return el.t;
	}

	// find seed to index and return handle
	// @param index position in Elements[] (THandle::Index)
	// @return invalid handle if index is out of range
	THandle GetHandleByIndex(uint32 index) const
	{
		THandle ret;
		if(index < Elements.size())
		{
			const TAndGuid& el = Elements[index];
			ret.Index = index;
			ret.Seed = el.Seed;
		}
		return ret;
	}

	// use like this:
	// for (uint32 it = 0, count = g_TextureManager.GetElementCount(); it < count; ++it)
	//     CTexHandle hnd = g_TextureManager.GetHandleByIterator(it);
	THandle GetHandleByIterator(uint32 Iterator) const
	{
		ASSERT(Iterator < UsedCount);

		uint32 ElementIndex = UsedAndFree[Iterator];

		const TAndGuid& el = Elements[ElementIndex];

		THandle ret;
		// Handle::Index must be the position in Elements[] (what GetElementPtr()/RemoveByIndex()
		// index with), matching push_back()'s Ret.Index = ElementIndex -- not el.IndexInUsed,
		// which (by this class's own invariant, see VerifyInternals()) always just equals
		// Iterator, the position in UsedAndFree/iteration order. The two only coincide by
		// accident when UsedAndFree still happens to be the identity permutation (nothing has
		// been removed-and-reused yet); any removal that shuffles the free list (SwapInUsedAndUsed)
		// exposes the difference, handing out a handle that points at the wrong element.
		ret.Index = ElementIndex;
		ret.Seed = el.Seed;
		return ret;
	}

	// for debugging
	uint32 GetDeadCount() const
	{
		return (uint32)Elements.size() - UsedCount - FreeCount;
	}

	void UpdateOrAddElement(THandle& outHnd, const T& data)
	{
		T* p = GetElementPtr(outHnd);

		if (p)
			*p = data;
		else
			outHnd = push_back(data);
	}

	// creates entry if needed, watch out, this might invalidate other pointers
	T* GetOrMakeElement(THandle hnd)
	{
		ASSERT(hnd.IsValid());
		T* p = GetElementPtr(hnd);

		if (!p)
		{
			Elements.resize(hnd.Index + 1);		// this might invalidate other pointers
			p = GetElementPtr(hnd);
		}

		return p;
	}

private:

	struct TAndGuid
	{
		// index into UsedAndFree to make RemoveElement O(1) instead of O(n)
		uint32 IndexInUsed;
		// incremented each time the index is reused (on remove), -1 if it cannot be reused any more
		// could be moved after "t" but this way it might be better for caching
		uint32 Seed;
		//
		T t;
	};

	std::vector<TAndGuid> Elements;

	// ranges:
	//   0 .. UsedCount-1								: used
	//   UsedCount .. UsedCount+FreeCount-1			: free
	//   UsedCount+FreeCount .. Elements.size()-1		: dead
	// elements: index into Elements[]
	std::vector<uint32> UsedAndFree;

	uint32 UsedCount;
	uint32 FreeCount;

	void SwapInUsedAndUsed(uint32 IndexA, uint32 IndexB)
	{
		if (IndexA != IndexB)
		{
			uint32& ElementIndexA = UsedAndFree[IndexA];
			uint32& ElementIndexB = UsedAndFree[IndexB];

			uint32& IndexInUsedA = Elements[ElementIndexA].IndexInUsed;
			uint32& IndexInUsedB = Elements[ElementIndexB].IndexInUsed;

			std::swap(IndexInUsedA, IndexInUsedB);
			std::swap(ElementIndexA, ElementIndexB);
		}
	}

	void VerifyInternals() const
	{
#ifdef _DEBUG
		uint32 Cnt = GetElementCount();

		for (uint32 Iterator = 0; Iterator < Cnt; ++Iterator)
		{
			const uint32 ElementIndex = UsedAndFree[Iterator];

			const TAndGuid& el = Elements[ElementIndex];

			ASSERT(el.IndexInUsed == Iterator);
		}
#endif
	}
};

// order can change when element get removed
// TElement can be some class derived from CObjHandle, having extra data store along the handle
template <class TElement = CObjHandle, int TSeedEnd = -1>
class CObjContainer
{
public:
	//
	void InitOnce(uint32 ReserveCount = 32)
	{
		HandleToPacked.reserve(1024);
		Packed.reserve(ReserveCount);

		VerifyInternals();
	}

	/*
		// O(1)
		void Add(const TElement& Hnd, bool bAssertIfNotUnique = true)
		{
			ASSERT(Hnd.IsValid());

			if (bAssertIfNotUnique)
			{
				// O(1)
				ASSERT(Find(Hnd) == -1);
			}
			SetIndexToPackedElements(Hnd, Packed.size());
			Packed.push_back(Hnd);

			VerifyInternals();
		}
	*/

	// if already in the container this simply updates the data
	// O(1)
	void AddOrUpdate(TElement Hnd)
	{
		ASSERT(Hnd.IsValid());

		VerifyInternals();

		int PackedIndex = Find(Hnd);

		if (PackedIndex == -1)
		{
			SetIndexToPackedElements(Hnd, (uint32)Packed.size());
			Packed.push_back(Hnd);
		}
		else
		{
			SetIndexToPackedElements(Hnd, PackedIndex);
			Packed[PackedIndex] = Hnd;
		}

		VerifyInternals();
	}

	// silent ignore if not in the container
	// O(1)
	void Remove(CObjHandle Hnd)
	{
		if (Hnd.IsValid())
		{
			VerifyInternals();

			int PackedIndex = Find(Hnd);

			if (PackedIndex != -1)
			{
				if (PackedIndex != Packed.size() - 1)
				{
					SetIndexToPackedElements(Packed.back(), PackedIndex);
					Packed[PackedIndex] = Packed.back();
				}
				Packed.pop_back();
				SetIndexToPackedElements(Hnd, -1);

				VerifyInternals();
			}
		}
	}

	/*// O(1), does not maintain the order
	// can be used with the Index from Iterate()
	void RemoveByIndex(uint32 PackedIndex)
	{
		std::vector<TElement>::iterator it = Packed.begin() + Index;

		if (it != Packed.end() - 1)
		{
			*it = Packed.back();
		}
		Packed.pop_back();

		VerifyInternals();
	}*/

	// O(1)
	// @return -1 if not found
	int Find(const CObjHandle& Hnd)
	{
		if (Hnd.Index < HandleToPacked.size())
		{
			int Index = HandleToPacked[Hnd.Index];

			if (Index != -1)
			{
				// the stored element could be a different one (different seed)
				if ((CObjHandle)Packed[Index] != Hnd)
				{
					Index = -1;
				}

				return Index;
			}
		}
		return -1;
	}

	// example usage:
	//  uint32 PackedIndex = 0;
	//  for (;;) {
	//    TElement& Hnd = Bullets.Iterate(PackedIndex);
	//    if (!Hnd.IsValid()) break;
	//    SElementData* El = Scene.GetElementPtr(Hnd);
	//    ...
	//  }
	//
	// usually O(1)
	// @param PackedIndex start with 0
	// @return invalid handle at the end
	TElement& Iterate(uint32& PackedIndex)
	{
		uint32 Size = (uint32)Packed.size();

		while (PackedIndex < Size)
		{
			TElement& Hnd = Packed[PackedIndex];

			//			T* Ret = Manager->GetElementPtr(Hnd);

			//			if (Ret)
			{
				++PackedIndex;
				return Hnd;
			}
			/*
						// fill hole with the last element
						if(Index != Size - 1)
						{
							Handles[Index] = Handles[Size - 1];
						}
						--Size;
						Handles.pop_back();
			*/
		}

		// reached the end
		static TElement Ret;
		// if this fails, someone manipulated the Null object returned from iteration
		ASSERT(!Ret.IsValid());

		return Ret;
	}

	void Clear()
	{
		HandleToPacked.clear();
		Packed.clear();

		VerifyInternals();
	}

	// for debugging, Note: iteration can change the count (invalid elements)
	uint32 GetCount() const
	{
		return (uint32)Packed.size();
	}

	// for debugging only
	void VerifyInternals() const
	{
#ifdef _DEBUG
		uint32 UsedInHandleToPacked = 0;
		{
			std::vector<uint32>::const_iterator it, end = HandleToPacked.end();

			uint32 HandleIndex = 0;
			for (it = HandleToPacked.begin(); it != end; ++it, ++HandleIndex)
			{
				int PackedIndex = *it;

				if (PackedIndex != -1)
				{
					++UsedInHandleToPacked;

					ASSERT(PackedIndex < (int)Packed.size());
					ASSERT(Packed[PackedIndex].Index == HandleIndex);
					ASSERT(Packed[PackedIndex].IsValid());
				}
			}
		}

		ASSERT(UsedInHandleToPacked == Packed.size());
#endif
	}

private:
	// [CObjHandle.Index]=Index in Packed[], -1 if not there
	std::vector<uint32> HandleToPacked;
	// packed CObjHandle with optional extra data
	std::vector<TElement> Packed;

	// only needed internally
	void SetIndexToPackedElements(const CObjHandle& Hnd, int PackedIndex)
	{
		ASSERT(Hnd.IsValid());

		if (Hnd.Index >= HandleToPacked.size())
		{
			HandleToPacked.resize(Hnd.Index + 1, -1);
		}

		HandleToPacked[Hnd.Index] = PackedIndex;
	}
};
void CObjManagerTest();
