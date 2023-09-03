/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../../algorithms/public/DCT.hpp"
#include "../../algorithms/public/MelBands.hpp"
#include "../../data/TensorTypes.hpp"
#include <cassert>

namespace fluid {
namespace algorithm {

class MFCC
{
public:
  MFCC(Allocator& alloc)
  {}

  void init(Allocator& alloc)
  {

  }

  void processFrame(Allocator& alloc)
  {

  }

private:

};
} // namespace algorithm
} // namespace fluid
