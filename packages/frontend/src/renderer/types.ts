import { Color3 } from '@babylonjs/core/Maths/math.color';

export interface SegbinData {
  count: number;
  geoms: Float32Array;
  roles: Uint8Array;
  chainContinue: Uint8Array;
  segType: Uint8Array;
  layerZs: Float32Array;
}

export const Role = {
  Perimeter:          0,
  ExternalPerimeter:  1,
  OverhangPerimeter:  2,
  InternalInfill:     3,
  SolidInfill:        4,
  TopSolidInfill:     5,
  BottomSurface:      6,
  BridgeInfill:       7,
  InternalBridgeInfill: 8,
  Travel:             9,
  SkirtBrim:          10,
  Support:            11,
  Ironing:            12,
  Other:              13,
} as const;

export type Role = (typeof Role)[keyof typeof Role];

export function roleLabel(r: number): string {
  return [
    'Perimeter', 'External Perimeter', 'Overhang Perimeter',
    'Internal Infill', 'Solid Infill', 'Top Solid Infill',
    'Bottom Surface', 'Bridge Infill', 'Internal Bridge Infill',
    'Travel', 'Skirt/Brim', 'Support', 'Ironing', 'Other',
  ][r] ?? 'Unknown';
}

export function roleColor(r: number): Color3 {
  const palette: Record<number, string> = {
    [Role.Perimeter]:              '#21749e',
    [Role.ExternalPerimeter]:      '#21749e',
    [Role.OverhangPerimeter]:      '#e1e1e1',
    [Role.InternalInfill]:         '#f0ff6c',
    [Role.SolidInfill]:            '#ae9e40',
    [Role.TopSolidInfill]:         '#b8303c',
    [Role.BottomSurface]:          '#ffa502',
    [Role.BridgeInfill]:           '#2ed573',
    [Role.InternalBridgeInfill]:   '#7bed9f',
    [Role.Travel]:                 '#000000',
    [Role.SkirtBrim]:              '#a4b0be',
    [Role.Support]:                '#33393c',
    [Role.Ironing]:                '#f8c291',
    [Role.Other]:                  '#f700ff',
  };
  return Color3.FromHexString(palette[r] ?? '#cccccc');
}
