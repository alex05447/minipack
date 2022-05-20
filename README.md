 # minipack

 A small utility library to create, access and patch resource/data file packs/archives.

 A "pack" is a folder which contains some arbitrary resource files used by the application.

 E.g., we may have a `"resources"` folder with some files in it.

 ```
 - resources
   |_ textures
   |  |_ chair
   |  |  |_ albedo.png
   |  |  |_ normal.tga
   |  |_ table
   |  |  |_ albedo.jpg
   |  |  |_ roughness.dds
   |_ meshes
   |  |_ chair.obj
   |  |_ table.fbx
   |_ shaders
   |  |_ base_pass.bin
   |  |_ lighting.bin
   |  |_ postprocess.bin
   |_ audio
      |_ music.ogg
      |_ sfx.wav
 ```

 Assume resource files are referred to by the application and accessed using their (UTF-8) relative paths in the `"resources"` folder.
 E.g., `"textures/chair/albedo.png"`, `"meshes/table.fbx"` (see `minifilepath`).

 A pack groups loose files in the source folder into one or more "data packs" (TODO: better name) files in order to

 1) minimize file system operations when deploying/patching - the less files the better;
 2) access the file contents more efficiently - by
     a) using path hashes instead of full string paths for lookup, and
     b) using memory mapping of a small number of data pack files instead of opening/reading a relatively large number of resource files;
 3) provide optional transparent compression.

 E.g., the above `"resources"` folder may be packed to a resource pack.
 It's a folder currently structured like this:

 ```
 - packed_resources
   |_ index
   |_ pack0
   |_ pack1
   |_ ...
   |_ pack<n>
   |_ strings
 ```

 The `"index"` file provides a lookup from the hash of the resource file's relative path in the `"resources"` folder (a simple `u64`, see `minifiletree::PathHash`)
 to its data, physically located in one of the "data pack" files (`"pack<n>"`).

 Number of "data packs" and source file data location within them should ideally be completely configurable,
 as it is very application specific.
 Having just one (potentially very large) "data pack" file is possible and legal, but this complicates the patching process
 (the way it currently works, patching process requires an amount of free disk space equal to the size of the largest "data pack" file),
 but this might be an accepatable tradeoff.
 Having multiple "data pack" files simplifies patching (by lowering the amount of free disk space required),
 and might synergize with some applications' resource usage patterns
 (e.g. having unique resources for each "map" / "level" in a separate "data pack", requiring only one "data pack" to be open / memory-mapped at a time).
 However, currently only the number of "data pack" files is configurable (via maximum file size setting),
 but not the resource file location within them. This might/needs to change in the future.

 Hashes of relative file paths (`minifiletree::PathHash`, a.k.a. `u64`) are used to refer to files in the pack instead of string paths for efficiency.
 I.e., hashes of paths like `"textures/chair/albedo.png"` and `"meshes/table.fbx"` (not quite simply string hashes of these, see `minifilepath` documentation).
 They may be calculated on demand or, ideally, precomputed at compile/resource cooking time.
 The hash function used for file path hashing is user-provided. Naturally, the same hash function needs to be used during lookups as the one used during packing.

 The (optional) `"strings"` file contains information necessary to map the file path hashes back to their UTF-8 relative path strings (see `minfiletree`).
 This is only necessary when one needs, e.g., to "unpack" the pack into the original folder/file hierarchy,
 or to visualize the original file tree structure in some sort of UI.

 This library allows to:

 1) create ("pack") a pack given a source directory path; with optional (multithreaded) compression of source file data.
 2) access the created pack, mapping file path hashes to file contents byte slices.
 3) (optionally) unpack a pack into its original file hierarchy.
 4) create a patch file given two source packs, "old" and "new".
 5) apply a patch file to the "old" pack in-place, resulting in a "new" pack.