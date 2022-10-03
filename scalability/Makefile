EXEC := ScalabilityTest

CPP := g++
CPPFLAGS := -O2 -Wall -std=c++17 -Isrc -I/usr/include/eigen3

LD := g++
LDFLAGS := -lfmt -lcasadi

SRCDIR := src
OBJDIR := build

# Make does not offer a recursive wildcard function, so here's one:
rwildcard=$(wildcard $1$2) $(foreach dir,$(wildcard $1*),$(call rwildcard,$(dir)/,$2))

SRC_CPP := $(call rwildcard,$(SRCDIR)/,*.cpp)
CPP_OBJ := $(addprefix $(OBJDIR)/,$(SRC_CPP:.cpp=.o))

.PHONY: all
all: $(OBJDIR)/$(EXEC)

-include $(CPP_OBJ:.o=.d)

$(OBJDIR)/$(EXEC): $(CPP_OBJ)
	@mkdir -p $(@D)
	$(LD) $+ $(LDFLAGS) -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CPP) $(CPPFLAGS) -MMD -c -o $@ $<

.PHONY: clean
clean:
	-$(RM) -r $(OBJDIR)
