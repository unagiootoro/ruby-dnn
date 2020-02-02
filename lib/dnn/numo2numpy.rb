# This library is a performs mutual conversion between Numo and Numpy.
# You need to install PyCall to use this library.
# [Usage]
# numpy to numo: Numpy.to_na(np_array)
# numo to numpy: Numpy.from_na(narray)

require "pycall/import"
require "numpy"

include PyCall::Import

class NumpyToNumoError < StandardError; end

module Numpy
  def self.from_na(narray)
    bin = narray.to_binary
    bin.force_encoding("ASCII-8BIT")
    case
    when narray.is_a?(Numo::Int8)
      Numpy.frombuffer(bin, dtype: "int8").reshape(*narray.shape)
    when narray.is_a?(Numo::UInt8)
      Numpy.frombuffer(bin, dtype: "uint8").reshape(*narray.shape)
    when narray.is_a?(Numo::Int16)
      Numpy.frombuffer(bin, dtype: "int16").reshape(*narray.shape)
    when narray.is_a?(Numo::UInt16)
      Numpy.frombuffer(bin, dtype: "uint16").reshape(*narray.shape)
    when narray.is_a?(Numo::Int32)
      Numpy.frombuffer(bin, dtype: "int32").reshape(*narray.shape)
    when narray.is_a?(Numo::UInt32)
      Numpy.frombuffer(bin, dtype: "uint32").reshape(*narray.shape)
    when narray.is_a?(Numo::Int64)
      Numpy.frombuffer(bin, dtype: "int64").reshape(*narray.shape)
    when narray.is_a?(Numo::UInt64)
      Numpy.frombuffer(bin, dtype: "uint64").reshape(*narray.shape)
    when narray.is_a?(Numo::SFloat)
      Numpy.frombuffer(bin, dtype: "float32").reshape(*narray.shape)
    when narray.is_a?(Numo::DFloat)
      Numpy.frombuffer(bin, dtype: "float64").reshape(*narray.shape)
    else
      raise NumpyToNumoError.new("#{narray.class.name} is not support convert.")
    end
  end

  def self.to_na(ndarray)
    shape = ndarray.shape
    bin = ndarray.flatten.tobytes
    case ndarray.dtype.to_s
    when "int8"
      Numo::Int8.from_binary(bin).reshape(*shape)
    when "uint8"
      Numo::UInt8.from_binary(bin).reshape(*shape)
    when "int16"
      Numo::Int16.from_binary(bin).reshape(*shape)
    when "uint16"
      Numo::UInt16.from_binary(bin).reshape(*shape)
    when "int32"
      Numo::Int32.from_binary(bin).reshape(*shape)
    when "uint32"
      Numo::UInt32.from_binary(bin).reshape(*shape)
    when "int64"
      Numo::Int64.from_binary(bin).reshape(*shape)
    when "uint64"
      Numo::UInt64.from_binary(bin).reshape(*shape)
    when "float32"
      Numo::SFloat.from_binary(bin).reshape(*shape)
    when "float64"
      Numo::DFloat.from_binary(bin).reshape(*shape)
    else
      raise NumpyToNumoError.new("#{ndarray.dtype} is not support convert.")
    end
  end
end
